import os
import math
import json
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
import re
from datetime import datetime
from typing import Tuple, Dict

try:
    from openai import AzureOpenAI

    OPENAI_OK = True
except Exception:
    OPENAI_OK = False
load_dotenv()
st.set_page_config(page_title="Whoz vs Thor Agent ", layout="wide")
st.title("Whoz vs Thor Agent (Discrepancy Checker)")
st.caption(
    "Compares Stage, Start_Date (>60d), Duration in months (>2), Probability. Names/owners shown but not compared.")

# --- Top banner (main area) ---
st.markdown(
    """
    <div style="
        background-color:#fff3cd; 
        color:#856404; 
        padding:12px; 
        border-radius:8px; 
        border:1px solid #ffeeba; 
        font-size:14px;
        margin-bottom: 12px;">
        ⚠️ <strong>Disclaimer:</strong> This is a <em>test/validation platform</em>. 
        The look and feel may vary in the actual production environment.<br><br>
        ⏳ <strong>Note:</strong> The agent may take longer to process discrepancies completely, 
        depending on the total number of records in Thor and Whoz. Please be patient while it is in progress.
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Database Connection")
    server = st.text_input("SQL Server", os.getenv("AZURE_SQL_SERVER"))
    database = st.text_input("Database", os.getenv("AZURE_SQL_DATABASE", ""))
    username = st.text_input("User", os.getenv("AZURE_SQL_USERNAME", ""), key="u")
    password = st.text_input("Password", os.getenv("AZURE_SQL_PASSWORD", ""), type="password")
    odbc_driver = st.text_input("ODBC Driver", os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server"))
    trust_cert = st.checkbox("Trust server certificate", value=True)
    max_rows = st.number_input("Max rows (preview)", min_value=1, max_value=200000, value=200000, step=50)

    st.divider()
    st.header("Rule Parameters")
    only_upcoming = st.checkbox("Filter to upcoming (Start ≥ tomorrow OR Task_Start ≥ tomorrow)", value=True)
    start_date_days_threshold = st.number_input("Start-Date discrepancy if > N days", min_value=0, max_value=365,
                                                value=60, step=1)
    duration_months_threshold = st.number_input("Duration discrepancy if > N months", min_value=0, max_value=24,
                                                value=2, step=1)

    st.divider()
    st.header("GenAI (optional)")
    use_llm = st.checkbox("Use GPT-4o mini for semantic Stage equivalence", value=False)
    llm_max_calls = st.number_input("Max LLM checks per run", min_value=0, max_value=10000, value=200, step=50)
    if use_llm and not OPENAI_OK:
        st.warning("`openai` package not found. Install `openai>=1.42.0`.")

DISPLAY_COLUMNS = [
    "na_sector", "Capability_L0", "delivery_unit", "CONTRACT_SIGN_DATE",
    "Opportunity_ID", "DOSSIER_EXTERNAL_ID",
    "Account_Name", "Dossier_Name",
    "Start_Date", "Task_Start_date",
    "Duration", "Whoz_Duration_Months",
    "Stage", "Dossier_Stage", "Dossier_stage_flow",
    "Probability", "Probability_Level",
    "Opportunity_Lead", "Dossier_Owner",
]

COLUMN_PAIRS = {
    "Start_Date": ("Start_Date", "Task_Start_date"),
    "Duration": ("Duration", "Whoz_Duration_Months"),
    "Stage": ("Stage", "Dossier_Stage"),
    "Probability": ("Probability", "Probability_Level"),
}


def get_engine(server, database, username, password, odbc_driver, trust_cert):
    url = URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=server,
        database=database,
        query={"driver": "ODBC Driver 17 for SQL Server",
               "Encrypt": "yes",
               "TrustServerCertificate": "no"},
    )
    return create_engine(url, fast_executemany=True, pool_pre_ping=True)


def clean_id(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        # remove leading OP# (any case) and optional space
        .str.replace(r'(?i)^op#\s*', '', regex=True)
        # in case there is plain "OP " in some rows
        .str.replace(r'(?i)^op\s*', '', regex=True)
    )


def get_rbac_info(engine, email: str) -> pd.DataFrame:
    q = text("""
       SELECT TOP 1
           email,
           distinct_role,
           L0,
           L1
       FROM dbo.rbac_info WITH (NOLOCK)
       WHERE LOWER(email) = LOWER(:email)
   """)
    return pd.read_sql_query(q, engine, params={"email": email})


def map_l1_to_delivery_unit(l1_value: str) -> str | None:
    if not l1_value:
        return None
    l1 = str(l1_value).strip().lower()
    # Mapping from your screenshot (RBAC L1 -> Sales Bookings DELIVERY_UNIT)
    mapping = {
        "frog innovation, strategy & de": "Invent NA - frog",
        "supply chain platform": "Invent NA - II - Supply Chain",
        "supply chain advisory": "Invent NA - II - Supply Chain",
        "business technology": "Invent NA - ET - Business Tech",
        "digital engineering & r&d": "Invent NA - II - Dig. Engineering",
        "change acceleration": "Invent NA - W&O - Change Accel.",
        "hr transformation": "Invent NA - W&O - HR Trans.",
        "data driven transformation": "Invent NA - DDT",
        "cx transformation": "Invent NA - frog",
        "corporate transformation": "Invent NA - ET - Corp. Trans.",
        "transformation management": "Invent NA - ET - Trans. Mgmt",
        "workforce transformation": "Invent NA - W&O - Workforce Trans.",

    }
    # “contains” match so truncated values like "frog Innovation, Strategy & De" still match
    for k, v in mapping.items():
        if k in l1:
            return v
    return None


def build_sales_rbac_filter(rbac_df: pd.DataFrame, user_email: str) -> Tuple[str, Dict, str]:
    """
    Returns:
      (sql_snippet_to_append_in_WHERE, params_dict, human_readable_desc)
    """
    # If user not found in RBAC -> return NO DATA (safe)
    if rbac_df is None or rbac_df.empty:
        return " AND 1=0 ", {}, "RBAC: email not found -> blocked"
    role = str(rbac_df.loc[0, "distinct_role"] or "").strip().lower()
    print(role)
    # st.write(f"Role: {role}")
    st.subheader(f"Role: {role}")
    l0 = str(rbac_df.loc[0, "L0"] or "").strip()
    l0_l = l0.lower()
    print(l0_l)
    # st.write(f"Industry: {l0_l}")
    st.subheader(f"CU/Ind: {l0_l}")
    l1 = str(rbac_df.loc[0, "L1"] or "").strip()
    # Decide if Industry vs Capability based on your rule
    is_industry = ("industry" in l0_l)
    print(is_industry)
    # Consider role check as you asked (CU / Ind. Lead)
    # (If you want strictly role-based gating, keep this)
    # role_ok = (role in {"CU/Ind. Lead ", "ind. lead", "ind lead", "ind.lead"}) or ("ind" in role and "lead" in role)
    role_ok = ("cu/ind. lead" in role)
    print(role_ok)
    # -------------------------
    # Industry filtering
    # -------------------------
    if role_ok and is_industry:
        # explicit mappings
        if "industry ls" in l0_l:
            return " AND NA_sector = :na_sector ", {"na_sector": "LS"}, "Industry LS -> NA_sector=LS"
        if "industry fs" in l0_l:
            return " AND NA_sector = :na_sector ", {"na_sector": "FS"}, "Industry FS -> NA_sector=FS"
        if "industry manufacturing & auto" in l0_l:
            return " AND NA_sector = :na_sector ", {"na_sector": "Auto",
                                                    "na_sector": "Manufacturing"}, "Industry Manufacturing & Auto -> NA_sector in (Manufacturing, Auto)"
        # fallback: try to take token after word 'industry'
        tokens = l0.replace("-", " ").split()
        idx = None
        for i, t in enumerate(tokens):
            if t.lower() == "industry":
                idx = i
                break
        if idx is not None and idx + 1 < len(tokens):
            sector = tokens[idx + 1].upper()
            return " AND NA_sector = :na_sector ", {"na_sector": sector}, f"Industry {sector} -> NA_sector={sector}"
        # If can't determine sector, block (or remove blocking if you prefer)
        return " AND 1=0 ", {}, f"Industry but sector not mapped from L0='{l0}' -> blocked"
    # -------------------------
    # Capability filtering
    # -------------------------
    # mapping examples you gave
    if role_ok:
        cap_map = {
            "synapse": "synapse",
            "frog customer first": "frog",
            "Intelligent Industry": "Intelligent Industry",
            "Enterprise Transformation": "ET",
            "Workforce & Organization": "W&O"

        }
        for k, v in cap_map.items():
            if k in l0_l:
                # NOTE: confirm exact column name in sales table.
                # You wrote: sales-Capabilities-lo column -> assumed as Sales_Capabilities_L0
                return " AND [Capability_L0] LIKE :cap_like ", {
                    "cap_like": f"%{v}%"}, f"Capability {k} -> Capability_L0 contains '{v}'"
        # If nothing matched, no extra filter (or block if you want strict RBAC)
        return " ", {}, f"No RBAC mapping matched for role='{role}', L0='{l0}'"

    # --- NEW: Practice Lead ---
    if role == "practice lead":
        delivery_unit = map_l1_to_delivery_unit(l1)
        # safer: if no mapping -> show nothing
        if not delivery_unit:
            return " AND 1=0 ", {}, f"Practice Lead but L1='{l1}' not mapped -> blocked"
        return (
            " AND [DELIVERY_UNIT] = :delivery_unit ",
            {"delivery_unit": delivery_unit},
            f"Practice Lead -> DELIVERY_UNIT='{delivery_unit}'"
        )

    # --- NEW: Default role -> filter by invent_spoc_email ---
    if role == "default":
        return (
            " AND LOWER(LTRIM(RTRIM(invent_spoc_email))) = LOWER(:user_email) ",
            {"user_email": user_email},
            f"Default role -> invent_spoc_email = {user_email}"
        )


def fetch_data(engine, max_rows: int, email: str):
    rbac_df = get_rbac_info(engine, email)
    rbac_sql, rbac_params, rbac_desc = build_sales_rbac_filter(rbac_df, email)
    st.sidebar.caption(f"RBAC user: {email}")
    st.sidebar.caption(f"{rbac_desc}")

    q_sales = f"""
        SELECT 
            Opportunity_ID,
            Account_Name,
            invent_spoc_email,
            CONTRACT_SIGN_DATE,
            Start_Date,
            Duration,
            Stage,
            na_sector,
            Capability_L0,
            delivery_unit,
            Probability,
            Opportunity_Lead
        FROM dbo.sales_bookings_scd WITH (NOLOCK)
        WHERE  Opportunity_ID IS NOT NULL and is_current =1  and Stage IN ('6 - proposing', '7 - formalising agreement','S - Sold') and FUNNEL_ALIGNMENT like '%Invent%'
        {rbac_sql}   
    """
    # and Stage NOT IN ('D - Dropped', 'L - Lost')
    print(q_sales)
    q_whoz = f"""
        SELECT 
            DOSSIER_EXTERNAL_ID,
            CONCAT('OP# ', DOSSIER_EXTERNAL_ID) as DOSSIER_EXTERNAL_ID_OP ,
            Dossier_Name,
            Task_Start_date,
            Task_End_date,
            Dossier_Stage,
            DOSSIER_SALES_WORKFLOW_STEP_NAME as Dossier_stage_flow,
            Dossier_probability AS Probability_Level,
            Dossier_Owner
        FROM dbo.GTD_WHOZ_DATA_scd WITH (NOLOCK)
        WHERE DOSSIER_EXTERNAL_ID IS NOT NULL  and is_current=1
    """
    # DOSSIER_SALES_WORKFLOW_STEP_NAME as stage_flow,
    df_sales = pd.read_sql_query(text(q_sales), engine, params=rbac_params)
    df_whoz = pd.read_sql_query(text(q_whoz), engine)

    df_sales["Opportunity_ID"] = clean_id(df_sales["Opportunity_ID"])
    df_whoz["DOSSIER_EXTERNAL_ID_OP"] = clean_id(df_whoz["DOSSIER_EXTERNAL_ID_OP"])

    df_sales["Opportunity_ID"] = df_sales["Opportunity_ID"].astype(str).str.strip()
    df_whoz["DOSSIER_EXTERNAL_ID_OP"] = df_whoz["DOSSIER_EXTERNAL_ID_OP"].astype(str).str.strip()

    def parse_date_col(series):
        return pd.to_datetime(series.astype(str).str.strip().replace({"": None, "nan": None}),
                              errors="coerce").dt.date

    df_sales["Start_Date"] = parse_date_col(df_sales["Start_Date"])
    df_sales["CONTRACT_SIGN_DATE"] = parse_date_col(df_sales["CONTRACT_SIGN_DATE"])
    df_whoz["Task_Start_date"] = parse_date_col(df_whoz["Task_Start_date"])
    df_whoz["Task_End_date"] = parse_date_col(df_whoz["Task_End_date"])

    days = (pd.to_datetime(df_whoz["Task_End_date"]) - pd.to_datetime(df_whoz["Task_Start_date"])).dt.days
    df_whoz["Whoz_Duration_Months"] = (days / 30.0).round().astype("Int64")

    def to_percent(x):
        if pd.isna(x): return math.nan
        s = str(x).strip().replace('%', '')
        try:
            return float(s)
        except:
            return math.nan

    df_sales["Probability"] = df_sales["Probability"].apply(to_percent)
    df_whoz["Probability_Level"] = df_whoz["Probability_Level"].apply(to_percent)

    return df_sales, df_whoz


def compare_rule_based(df_sales: pd.DataFrame, df_whoz: pd.DataFrame,
                       start_days_thresh: int, dur_months_thresh: int, upcoming: bool,
                       use_llm_flag: bool, max_llm_calls: int):
    # ✅ Create the concatenated column to match OPPORTUNITY_ID format
    # df_whoz["DOSSIER_EXTERNAL_ID_OP"] = "OP# " + df_whoz["DOSSIER_EXTERNAL_ID"].astype(str).str.zfill(7)
    merged = df_sales.merge(
        df_whoz,
        how="inner",
        left_on="Opportunity_ID",
        right_on="DOSSIER_EXTERNAL_ID_OP",
        suffixes=("_Thor", "_Whoz")
    )
    merged = merged.drop_duplicates(keep='first')

    for col in DISPLAY_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[DISPLAY_COLUMNS].copy()
    # st.write(merged, unsafe_allow_html=True)
    # if upcoming:
    #    tomorrow = pd.Timestamp(date.today() + timedelta(days=1)).date()
    #    print(tomorrow)
    #    # tomorrow = pd.Timestamp("2025-05-01").date()
    #    s = pd.to_datetime(merged["Start_Date"], errors="coerce").dt.date
    #    w = pd.to_datetime(merged["Task_Start_date"], errors="coerce").dt.date
    #    merged = merged[(s >= tomorrow) | (w >= tomorrow)].copy()
    #    st.write(merged, unsafe_allow_html=True)
    if upcoming:
        today = pd.Timestamp(date.today()).date()
        end_date = (pd.Timestamp(date.today()) + timedelta(days=60)).date()
        # Convert ContractSignDate to date
        c = pd.to_datetime(
            merged["CONTRACT_SIGN_DATE"],
            errors="coerce"
        ).dt.date
        # Filter between today and today + 60 days
        merged = merged[
            (c >= today) & (c <= end_date)
            ].copy()
        # st.write(merged, unsafe_allow_html=True)

    # --- NEW STAGE LOGIC ---

    def _norm(s: pd.Series) -> pd.Series:
        return s.fillna("").str.strip().str.lower()

    dossier_stage = _norm(merged["Dossier_Stage"])
    # 👉 use the exact column name for the flow column from your dataframe
    dossier_flow = _norm(merged["Dossier_stage_flow"])  # change name if needed
    thor_stage = _norm(merged["Stage"])
    # Start with all False (no mismatch)
    stage_mismatch = pd.Series(False, index=merged.index)
    # 1) When dossier stage is OPEN
    mask_open = dossier_stage == "open"
    # 1a) These flows must match specific Thor stages
    flow_to_thor = {
        "thor stage 6: proposing": "6 - proposing",
        "thor stage 7: formalising agreement": "7 - formalising agreement",
        "thor stage 5: finalising the solution": "5 - finalising the solution",
    }

    for flow_val, thor_expected in flow_to_thor.items():
        mask = mask_open & (dossier_flow == flow_val)
        # mismatch if Thor stage is NOT the expected one
        stage_mismatch[mask] = thor_stage[mask] != thor_expected
    # 1b) For OPEN: Negotiation / Proposal / Qualification are always invalid
    invalid_flows_open = {"negotiation", "proposal", "qualification"}
    mask_invalid_open = mask_open & dossier_flow.isin(invalid_flows_open)
    stage_mismatch[mask_invalid_open] = True  # always flag
    # 2) When dossier stage is CLOSED_WON: Thor stage MUST be 'S - Sold'
    mask_closed_won = dossier_stage == "closed_won"
    stage_mismatch[mask_closed_won] = thor_stage[mask_closed_won] != "s - sold"
    # 3) (Optional) for any other combinations, fall back to simple equality
    mask_other = ~(mask_open | mask_closed_won)
    stage_mismatch[mask_other] = thor_stage[mask_other] != dossier_stage[mask_other]
    merged["Stage_Mismatch"] = stage_mismatch

    # --- END NEW STAGE LOGIC ---

    def to_percent(x):
        x = pd.to_numeric(x, errors="coerce")
        return x * 100 if x <= 1 else x

    merged["prob1_norm"] = merged["Probability"].apply(to_percent)
    merged["prob2_norm"] = merged["Probability_Level"].apply(to_percent)
    merged["Probability_Mismatch"] = merged["prob1_norm"] != merged["prob2_norm"]

    a = pd.to_datetime(merged["Start_Date"], errors="coerce")
    b = pd.to_datetime(merged["Task_Start_date"], errors="coerce")
    diff_days = (a - b).abs().dt.days.fillna(0)
    merged["Start_Date_DiffDays"] = diff_days
    merged["Start_Date_Mismatch"] = diff_days > start_days_thresh

    d_left = pd.to_numeric(merged["Duration"], errors="coerce")
    d_right = pd.to_numeric(merged["Whoz_Duration_Months"], errors="coerce")
    dur_diff = (d_left - d_right).abs()
    merged["Duration_DiffMonths"] = dur_diff
    merged["Duration_Mismatch"] = dur_diff > dur_months_thresh

    final_cols = [
        "Opportunity_ID", "DOSSIER_EXTERNAL_ID",
        "Account_Name", "Dossier_Name",
        "Start_Date", "Task_Start_date", "Start_Date_DiffDays", "Start_Date_Mismatch",
        "Duration", "Whoz_Duration_Months", "Duration_DiffMonths", "Duration_Mismatch",
        "Stage", "Dossier_Stage", "Dossier_stage_flow", "Stage_Mismatch",
        "Probability", "Probability_Level", "Probability_Mismatch",
        "Opportunity_Lead", "Dossier_Owner",
    ]
    # Keep only columns that actually exist (robust if some flags not present)
    final_cols = [c for c in final_cols if c in merged.columns]
    merged = merged[final_cols].copy()

    # Build a style dataframe *aligned with the final order*
    def highlight_pairs(df: pd.DataFrame):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        red = "background-color:#ffcccc;font-weight:600;"
        if "Start_Date_Mismatch" in df.columns:
            styles.loc[df["Start_Date_Mismatch"] == True, ["Start_Date", "Task_Start_date"]] = red
        if "Duration_Mismatch" in df.columns:
            styles.loc[df["Duration_Mismatch"] == True, ["Duration", "Whoz_Duration_Months"]] = red
        if "Stage_Mismatch" in df.columns:
            styles.loc[df["Stage_Mismatch"] == True, ["Stage", "Dossier_Stage"]] = red
        if "Probability_Mismatch" in df.columns:
            styles.loc[df["Probability_Mismatch"] == True, ["Probability", "Probability_Level"]] = red
        return styles

    styled = merged.style.apply(highlight_pairs, axis=None)
    return merged, styled


def enrich_with_owner_email(
        df: pd.DataFrame, engine, dossier_col: str = "DOSSIER_EXTERNAL_ID"
) -> pd.DataFrame:
    out = df.copy()

    # --- Clean IDs robustly: drop NaN BEFORE astype(str) ---
    s = out[dossier_col]
    s = s[s.notna()]  # drop actual NaN values
    s = s.astype(str).str.strip()
    # Exclude empty strings and common string-NA sentinels
    s = s[s.ne("") & s.str.lower().ne("nan") & s.str.lower().ne("none")]
    ids = s.unique().tolist()

    if not ids:
        out["CAPGEMINI_EMAIL"] = pd.NA
        return out

    # Build parameterized VALUES list
    values_rows_sql = ",\n                ".join(
        [f"(CAST(:id{i} AS NVARCHAR(100)))" for i in range(len(ids))]
    )

    sql = text(f"""
        ;WITH input_ids AS (
            SELECT v.DOSSIER_EXTERNAL_ID_IN
            FROM (
                VALUES
                    {values_rows_sql}
            ) AS v(DOSSIER_EXTERNAL_ID_IN)
        )
        SELECT DISTINCT
            ii.DOSSIER_EXTERNAL_ID_IN AS DOSSIER_EXTERNAL_ID,
            h.CAPGEMINI_EMAIL
        FROM input_ids AS ii
        JOIN dbo.GTD_WHOZ_DATA_SCD AS g WITH (NOLOCK)
          ON g.DOSSIER_EXTERNAL_ID LIKE '%' + ii.DOSSIER_EXTERNAL_ID_IN + '%'
         AND g.DOSSIER_OWNER_EXTERNAL_ID IS NOT NULL
         AND g.IS_CURRENT = 1
        LEFT JOIN dbo.HEADCOUNT AS h WITH (NOLOCK)   -- LEFT JOIN for diagnostics; change back to JOIN if required
          ON LTRIM(RTRIM(CAST(h.GLOBAL_ID AS NVARCHAR(50)))) =
             LTRIM(RTRIM(CAST(g.DOSSIER_OWNER_EXTERNAL_ID AS NVARCHAR(50))));
    """)

    params = {f"id{i}": ids[i] for i in range(len(ids))}

    mapping = pd.read_sql_query(sql, engine, params=params)

    # If no rows returned at all
    if mapping.empty:
        out["CAPGEMINI_EMAIL"] = pd.NA
        return out

    # --- Normalize keys on BOTH sides consistently ---
    # Create a normalized helper key (preserves the original column for output)
    out["_merge_key_"] = (
        out[dossier_col].astype(str).str.strip().str.upper()
    )
    mapping["_merge_key_"] = (
        mapping["DOSSIER_EXTERNAL_ID"].astype(str).str.strip().str.upper()
    )

    # If multiple mapping rows for a single key, pick one deterministically
    mapping = (
        mapping.sort_values(["_merge_key_", "CAPGEMINI_EMAIL"], na_position="last")
        .groupby("_merge_key_", as_index=False)
        .first()
    )

    # Merge on the normalized helper key
    out = out.merge(mapping[["_merge_key_", "CAPGEMINI_EMAIL"]],
                    how="left", on="_merge_key_").drop(columns=["_merge_key_"])

    # Ensure column exists
    if "CAPGEMINI_EMAIL" not in out.columns:
        out["CAPGEMINI_EMAIL"] = pd.NA

    return out


# #mail_variable = "dylan@synapse.com"
# mail_variable = "michele.pesanello@capgemini.com"
# #practice lead mail
# mail_variable = "tom-xyz@gmail.com"
# if st.button("🔎 Explore (Pandas parses dates)", use_container_width=True):
#     with st.spinner("Connecting and fetching…"):
#         try:
#             engine = get_engine(server, database, username, password, odbc_driver, trust_cert)
#             df_sales, df_whoz = fetch_data(engine, int(max_rows), mail_variable)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.stop()

#     st.success(f"Fetched {len(df_sales)} Thor rows and {len(df_whoz)} Whoz rows.")
#     df, styled = compare_rule_based(df_sales, df_whoz,
#                                     int(start_date_days_threshold),
#                                     int(duration_months_threshold),
#                                     bool(only_upcoming),
#                                     bool(use_llm),
#                                     int(llm_max_calls))

#     st.subheader("Discrepancy Table (red = discrepancy)")
#     st.write(df, unsafe_allow_html=True)
#     st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
#                        file_name="whoz_thor_discrepancies_sqlalchemy_pandas_dates.csv", mime="text/csv")
# else:
#     st.info("Fill DB details in the sidebar and click **Explore (Pandas parses dates)**.")

# mail_variable = "dennis.ephlin@capgemini.com"
# engine = get_engine(server, database, username, password, odbc_driver, trust_cert)
# df_sales, df_whoz = fetch_data(engine, mail_variable)
# df, styled = compare_rule_based(df_sales, df_whoz,
#                                     int(start_date_days_threshold),
#                                     int(duration_months_threshold),
#                                     bool(only_upcoming))
# df.to_csv("test_whozvsThor_without_email_02.csv", index = False)
# df_enriched = enrich_with_owner_email(df, engine, dossier_col="DOSSIER_EXTERNAL_ID")

# df_enriched.to_csv("test_whozvsThor_with_email_02.csv", index = False)


# print(df_enriched)

# --- UI: Email input ---
# mail_variable = st.text_input(
#     "Email (Whoz/Thor filter)",
#     value="tom-xyz@gmail.com",     # default value (optional)
#     placeholder="name@example.com",
#     key="mail_input"
# ).strip()

# # Optional: basic email validation (simple regex)
# def is_valid_email(s: str) -> bool:
#     return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s))

# if st.button("🔎 Check Discrepancy ", use_container_width=True):
#     if not mail_variable:
#         st.warning("Please enter an email.")
#         st.stop()
#     if not is_valid_email(mail_variable):
#         st.error("Please enter a valid email address.")
#         st.stop()

#     with st.spinner("Connecting and fetching…"):
#         try:
#             engine = get_engine(server, database, username, password, odbc_driver, trust_cert)
#             # Pass the email from UI here ⤵
#             df_sales, df_whoz = fetch_data(engine, int(max_rows), mail_variable)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.stop()

#     st.success(f"Fetched {len(df_sales)} Thor rows and {len(df_whoz)} Whoz rows.")
#     df, styled = compare_rule_based(
#         df_sales, df_whoz,
#         int(start_date_days_threshold),
#         int(duration_months_threshold),
#         bool(only_upcoming),
#         bool(use_llm),
#         int(llm_max_calls)
#     )

#     df_enriched = enrich_with_owner_email(df, engine, dossier_col="DOSSIER_EXTERNAL_ID")

#     st.subheader("Discrepancy Table (red = discrepancy)")
#     st.write(df_enriched, unsafe_allow_html=True)
#     st.download_button(
#         "⬇️ Download CSV",
#         df.to_csv(index=False).encode("utf-8"),
#         file_name="whoz_thor_discrepancies_sqlalchemy_pandas_dates.csv",
#         mime="text/csv"
#     )
# else:
#     st.info("Fill DB details in the sidebar and click **Explore (Pandas parses dates)**.")


mail_variable = st.text_input(
    "Email (Whoz/Thor filter)",
    value="tom-xyz@gmail.com",  # default value (optional)
    placeholder="name@example.com",
    key="mail_input"
).strip()


# Optional: basic email validation (simple regex)
def is_valid_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s))


# Turn an email into a filesystem-safe filename component
def email_to_filename(email: str, prefix="whoz_thor_discrepancies", ext="csv", add_timestamp=True) -> str:
    if not email:
        safe_email = "unknown"
    else:
        e = email.strip().lower()
        e = e.replace("@", "_at_").replace(".", "_")
        safe_email = re.sub(r"[^a-z0-9_\-]+", "_", e).strip("_")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None
    if ts:
        return f"{safe_email}_{prefix}_{ts}.{ext}"
    return f"{safe_email}_{prefix}.{ext}"


if st.button("🔎 Check Discrepancy ", use_container_width=True):
    if not mail_variable:
        st.warning("Please enter an email.")
        st.stop()
    if not is_valid_email(mail_variable):
        st.error("Please enter a valid email address.")
        st.stop()

    with st.spinner("Connecting and fetching…"):
        try:
            engine = get_engine(server, database, username, password, odbc_driver, trust_cert)
            # Pass the email from UI here ⤵
            df_sales, df_whoz = fetch_data(engine, int(max_rows), mail_variable)
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # st.success(f"Fetched {len(df_sales)} Thor rows and {len(df_whoz)} Whoz rows.")
    df, styled = compare_rule_based(
        df_sales, df_whoz,
        int(start_date_days_threshold),
        int(duration_months_threshold),
        bool(only_upcoming),
        bool(use_llm),
        int(llm_max_calls)
    )

    # Enrich after comparison
    df_enriched = enrich_with_owner_email(df, engine, dossier_col="DOSSIER_EXTERNAL_ID")

    st.subheader("Discrepancy Table (red = discrepancy)")
    # If you want scrollable, use st.dataframe(df_enriched); st.write works fine too
    st.write(df_enriched, unsafe_allow_html=True)

    # Build email-based filename
    filename = email_to_filename(mail_variable)

    st.download_button(
        "⬇️ Download CSV",
        df_enriched.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )
else:
    st.info("Fill DB details in the sidebar and click **Explore (Pandas parses dates)**.")

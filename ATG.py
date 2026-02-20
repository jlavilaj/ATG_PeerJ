
from typing import Optional

import pandas as pd
from stellargraph import StellarDiGraph


def create_atg_graph(
    student_data: pd.DataFrame,
    *,
    student_id_col: str = "STD_STUDENT_ID",
    semester_col: str = "CRS_SEMESTER_DATE",
    course_id_col: str = "CRS_CRN_COURSEREFNO",
    upto_semester: Optional[object] = None,
) -> StellarDiGraph:
    """
    Build an Academic Trajectory Graph (ATG) for a single student.

    The ATG is a homogeneous StellarDiGraph containing:
      - one student node:        "S_<student_id>"
      - course-enrolment nodes:  "C_<course_id>__<semester>__<rep>"

    Edges:
      - student -> course (enrolment)
      - course(t) -> course(t+1) for consecutive semesters (transitions)

    Leakage-safe by design:
      - No GPA, grades, cumulative indicators, or outcome-derived variables are used.
      - Optionally restrict the graph to the prefix up to `upto_semester` (inclusive).

    Parameters
    ----------
    student_data : pd.DataFrame
        Records for ONE student (potentially multiple semesters).
    upto_semester : optional
        Cutoff semester (inclusive). If provided, uses only rows with
        semester_col <= upto_semester.

    Returns
    -------
    StellarDiGraph
    """
    required = {student_id_col, semester_col, course_id_col}
    missing = required.difference(student_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = student_data.copy().sort_values(semester_col)

    # Optional: keep only information available up to semester t (inclusive)
    if upto_semester is not None:
        df = df[df[semester_col] <= upto_semester].copy()

    if df.empty:
        raise ValueError("No records left after applying upto_semester filter.")

    # -------------------------
    # 1) Student node id
    # -------------------------
    student_id = str(df[student_id_col].iloc[0])
    student_node_id = f"S_{student_id}"

    # -------------------------
    # 2) Unique course node ids (handle duplicates within same semester)
    # -------------------------
    df["rep"] = df.groupby([course_id_col, semester_col]).cumcount()
    df["course_node_id"] = (
        "C_" + df[course_id_col].astype(str)
        + "__" + df[semester_col].astype(str)
        + "__" + df["rep"].astype(str)
    )

    # -------------------------
    # 3) Course node features (leakage-safe, schedule/structure only)
    # -------------------------
    candidate_course_features = [
        "CRS_CREDIT_HOURS",
        "CRS_SUNDAY_WEIGHT", "CRS_MONDAY_WEIGHT", "CRS_TUESDAY_WEIGHT",
        "CRS_WENDESDAY_WEIGHT", "CRS_THURESDAY_WEIGHT",
        "CRS_FRIDAY_WEIGHT", "CRS_SATURDAY_WEIGHT",
        "CRS_SUMDAYS_WEIGHTS",
        "CRS_TIME_EARLYMORNING_WEIGHT", "CRS_TIMEMORNING_WEIGHT",
        "CRS_TIMEAFTERNOON_WEIGHT", "CRS_TIMEEVENING_WEIGHT",
        "CRS_TIMENIGHT_WEIGHT",
        "CRS_SUMTIMES_WEIGHTS",
    ]
    course_feature_cols = [c for c in candidate_course_features if c in df.columns]

    course_nodes = (
        df.drop_duplicates("course_node_id")
          .set_index("course_node_id")[course_feature_cols]
          .apply(pd.to_numeric, errors="coerce")
          .fillna(0.0)
          .astype(float)
    )

    # -------------------------
    # 4) Student node features (structural only)
    # -------------------------
    student_nodes = pd.DataFrame(
        {
            "num_unique_courses": [float(df[course_id_col].nunique())],
            "num_enrolments": [float(len(df))],
            "num_semesters_observed": [float(df[semester_col].nunique())],
        },
        index=[student_node_id],
    ).astype(float)

    # -------------------------
    # 5) Align feature dimensions for homogeneous graph
    # -------------------------
    for col in student_nodes.columns:
        if col not in course_nodes.columns:
            course_nodes[col] = 0.0
    for col in course_nodes.columns:
        if col not in student_nodes.columns:
            student_nodes[col] = 0.0

    # Stable column order
    course_nodes = course_nodes.sort_index(axis=1)
    student_nodes = student_nodes[course_nodes.columns]

    all_nodes = pd.concat([student_nodes, course_nodes], axis=0)

    # -------------------------
    # 6) Edges
    # -------------------------
    # student -> courses
    enrolled_edges = (
        pd.DataFrame({"source": student_node_id, "target": df["course_node_id"].astype(str)})
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # course(t) -> course(t+1) (dense bipartite between consecutive semesters)
    semesters = list(pd.unique(df[semester_col]))
    src, tgt = [], []

    for s_now, s_next in zip(semesters[:-1], semesters[1:]):
        c_now = df.loc[df[semester_col] == s_now, "course_node_id"].astype(str).unique()
        c_next = df.loc[df[semester_col] == s_next, "course_node_id"].astype(str).unique()
        for a in c_now:
            for b in c_next:
                src.append(a)
                tgt.append(b)

    transition_edges = pd.DataFrame({"source": src, "target": tgt}).drop_duplicates()

    all_edges = pd.concat([enrolled_edges, transition_edges], axis=0).reset_index(drop=True)

    # -------------------------
    # 7) Build homogeneous graph
    # -------------------------
    return StellarDiGraph(nodes=all_nodes, edges=all_edges)    
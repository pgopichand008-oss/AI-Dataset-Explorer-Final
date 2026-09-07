"""
test_suite.py — Comprehensive automated test suite for AI Dataset Intelligence Platform.
Executes all 18 mandatory test cases specified in the project requirements.
"""

from __future__ import annotations
import sys
import os

# Set UTF-8 stdout encoding for Windows console compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import unittest
import pandas as pd
import numpy as np

# Engine & Agent imports
from change_engine import compare_datasets, compare_statistics
from quality_engine import compare_quality
from engine.ai_agent import (
    run_intelligence_analysis,
    reconsider_previous_finding,
    assess_impact,
    assess_ml_readiness,
    generate_recommendation,
    determine_next_action,
    build_step_statuses,
)
from engine.visualization import (
    create_3d_scatter,
    create_3d_surface,
    create_3d_mesh,
    recommend_visualizations,
    get_graph_guide,
)
from engine.calculation_engine import perform_calculation
from engine.ai_report import generate_ai_report, _local_fallback_report
from utils import prepare_safe_dataframe, validate_chart_inputs


class TestAIDatasetIntelligence(unittest.TestCase):

    def setUp(self):
        # Base normal dataset
        self.df_base = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "income": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
            "score": [80.5, 85.0, 90.2, 75.4, 88.6, 92.1, 84.3, 79.0, 95.5, 87.2],
            "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        })

    def test_1_normal_dataset(self):
        """TEST 1: Normal dataset change comparison (identical data)."""
        res = compare_datasets(self.df_base, self.df_base.copy())
        self.assertEqual(len(res["added_columns"]), 0)
        self.assertEqual(len(res["removed_columns"]), 0)
        self.assertEqual(len(res["type_changes"]), 0)
        self.assertEqual(len(res["possible_renames"]), 0)
        print("[PASS] TEST 1: Normal dataset comparison")

    def test_2_added_column(self):
        """TEST 2: Detect added column."""
        df_new = self.df_base.copy()
        df_new["signup_date"] = "2026-01-01"
        res = compare_datasets(self.df_base, df_new)
        self.assertIn("signup_date", res["added_columns"])
        print("[PASS] TEST 2: Added column detected")

    def test_3_removed_column(self):
        """TEST 3: Detect removed column."""
        df_new = self.df_base.drop(columns=["score"])
        res = compare_datasets(self.df_base, df_new)
        self.assertIn("score", res["removed_columns"])
        print("[PASS] TEST 3: Removed column detected")

    def test_4_possible_rename(self):
        """TEST 4: Detect possible column rename."""
        df_old = pd.DataFrame({"customer_age": [20, 30, 40, 50], "val": [1, 2, 3, 4]})
        df_new = pd.DataFrame({"age": [20, 30, 40, 50], "val": [1, 2, 3, 4]})
        res = compare_datasets(df_old, df_new)
        renames = res["possible_renames"]
        self.assertTrue(len(renames) > 0)
        self.assertEqual(renames[0]["old_column"], "customer_age")
        self.assertEqual(renames[0]["new_column"], "age")
        self.assertIn("reason", renames[0])
        print("[PASS] TEST 4: Column rename detected with explainable reason")

    def test_5_type_conflict(self):
        """TEST 5: Detect high-severity type conflict."""
        df_old = pd.DataFrame({"income": [50000, 60000, 70000]})
        df_new = pd.DataFrame({"income": ["ABC", "XYZ", "DEF"]})
        res = compare_datasets(df_old, df_new)
        type_changes = res["type_changes"]
        self.assertTrue(len(type_changes) > 0)
        self.assertEqual(type_changes[0]["conflict_level"], "HIGH")
        print("[PASS] TEST 5: High-severity type conflict detected")

    def test_6_mixed_valid_invalid_values(self):
        """TEST 6: Mixed valid/invalid values handling."""
        df_mixed = pd.DataFrame({"amount": ["100", "200", "unknown", "300", "N/A"]})
        res = compare_datasets(self.df_base, df_mixed)
        inv = res["invalid_values"]
        self.assertIn("amount", inv)
        self.assertEqual(inv["amount"]["invalid_count"], 2)
        self.assertEqual(inv["amount"]["valid_count"], 3)
        print("[PASS] TEST 6: Mixed valid/invalid values safely reported")

    def test_7_increased_missing_values(self):
        """TEST 7: Detect increased missing values."""
        df_new = self.df_base.copy()
        df_new.loc[0:4, "income"] = np.nan
        q_res = compare_quality(self.df_base, df_new)
        chg = [item for item in q_res["missing_changes"] if item["column"] == "income"]
        self.assertTrue(len(chg) > 0)
        self.assertTrue(chg[0]["change"] > 0)
        print("[PASS] TEST 7: Increased missing values detected")

    def test_8_statistical_change(self):
        """TEST 8: Statistical change calculation."""
        df_new = self.df_base.copy()
        df_new["income"] = df_new["income"] * 2.5
        stats = compare_statistics(self.df_base, df_new)
        inc_stat = [s for s in stats if s["column"] == "income"][0]
        self.assertTrue(inc_stat["mean_change"] > 0)
        print("[PASS] TEST 8: Statistical change accurately calculated")

    def test_9_previous_finding_reconsideration(self):
        """TEST 9: Previous finding reconsideration required."""
        changes = {"removed_columns": ["age"], "type_changes": [], "invalid_values": {}}
        quality = {"missing_changes": []}
        finding = "Revenue is strongly related to customer age."
        recon = reconsider_previous_finding(finding, changes, quality)
        self.assertEqual(recon["status"], "RECONSIDER")
        self.assertTrue(len(recon["warnings"]) > 0)
        print("[PASS] TEST 9: Reconsideration triggered when removed column impacts finding")

    def test_10_no_previous_finding(self):
        """TEST 10: No previous finding provided."""
        changes = {"removed_columns": [], "type_changes": []}
        quality = {"missing_changes": []}
        recon = reconsider_previous_finding("", changes, quality)
        self.assertEqual(recon["status"], "NO_PREVIOUS_FINDING")
        print("[PASS] TEST 10: Empty finding handled gracefully")

    def test_11_ml_readiness(self):
        """TEST 11: Explainable ML readiness assessment."""
        changes = {"type_changes": [{"column": "income", "conflict_level": "HIGH"}], "invalid_values": {}}
        quality = {"missing_changes": []}
        ml_res = assess_ml_readiness(changes, quality)
        self.assertEqual(ml_res["status"], "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION")
        print("[PASS] TEST 11: ML readiness status evaluated deterministically")

    def test_12_complete_workflow(self):
        """TEST 12: Complete 6-stage Mystery Mission workflow execution."""
        res = run_intelligence_analysis(
            self.df_base,
            self.df_base,
            previous_finding="Age correlates with income."
        )
        self.assertEqual(len(res["step_statuses"]), 6)
        self.assertEqual(res["step_statuses"][0]["step"], "DETECT")
        self.assertEqual(res["step_statuses"][5]["step"], "REPORT")
        print("[PASS] TEST 12: Complete 6-stage workflow executed cleanly")

    def test_13_3d_visualization_three_numeric(self):
        """TEST 13: 3D visualization with 3 numeric columns."""
        fig, meta = create_3d_scatter(self.df_base, "age", "income", "score")
        self.assertIsNotNone(fig)
        self.assertEqual(meta["valid_rows"], 10)
        self.assertEqual(meta["excluded_rows"], 0)
        print("[PASS] TEST 13: 3D scatter plot generated from 3 numeric columns")

    def test_14_3d_visualization_fewer_than_three_numeric(self):
        """TEST 14: 3D visualization unavailable message check (<3 numeric cols)."""
        df_small = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        recs = recommend_visualizations(df_small)
        three_d_recs = [r for r in recs if r["type"] == "3D Scatter"]
        self.assertEqual(len(three_d_recs), 0)
        print("[PASS] TEST 14: 3D scatter suppressed when <3 numeric columns exist")

    def test_15_3d_visualization_missing_values(self):
        """TEST 15: 3D visualization handling missing values."""
        df_missing = self.df_base.copy()
        df_missing.loc[0, "age"] = np.nan
        fig, meta = create_3d_scatter(df_missing, "age", "income", "score")
        self.assertIsNotNone(fig)
        self.assertEqual(meta["valid_rows"], 9)
        self.assertEqual(meta["excluded_rows"], 1)
        print("[PASS] TEST 15: 3D scatter filtered NaNs and reported observation counts")

    def test_16_3d_visualization_invalid_values(self):
        """TEST 16: 3D visualization handling invalid numeric strings."""
        df_invalid = self.df_base.copy()
        df_invalid["score"] = df_invalid["score"].astype(object)
        df_invalid.loc[0, "score"] = "invalid_number"
        fig, meta = create_3d_scatter(df_invalid, "age", "income", "score")
        self.assertIsNotNone(fig)
        self.assertEqual(meta["valid_rows"], 9)
        self.assertEqual(meta["excluded_rows"], 1)
        print("[PASS] TEST 16: 3D scatter safely coerced invalid numbers")

    def test_17_visualization_selector(self):
        """TEST 17: Intelligent visualization selector recommendations."""
        recs = recommend_visualizations(self.df_base)
        self.assertTrue(len(recs) >= 3)
        self.assertTrue(any(r["title"] == "3D Scatter Visualization" for r in recs))
        self.assertTrue(any(r["title"] == "Correlation Heatmap" for r in recs))
        print("[PASS] TEST 17: Intelligent visualization selector recommends appropriate charts")

    def test_18_gemini_unavailable_fallback(self):
        """TEST 18: Deterministic report fallback when Gemini API unavailable."""
        report = _local_fallback_report(
            profile={"rows": 100, "columns": 5, "missing_cells": 0, "duplicate_rows": 0},
            quality_findings=["High cardinality in customer_id"],
            quality_score=95,
            ml_readiness=90,
        )
        self.assertIn("# Executive Assessment", report)
        self.assertIn("# Final Verdict", report)
        self.assertIn("🟢 READY FOR FURTHER ANALYSIS", report)
        print("[PASS] TEST 18: Deterministic fallback report operates cleanly without Gemini")

    def test_19_descriptive_stats_accuracy(self):
        """TEST 19: Mathematical accuracy check for descriptive statistics."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(float(s.mean()), 30.0)
        self.assertAlmostEqual(float(s.median()), 30.0)
        self.assertAlmostEqual(float(s.min()), 10.0)
        self.assertAlmostEqual(float(s.max()), 50.0)
        self.assertAlmostEqual(float(s.quantile(0.75) - s.quantile(0.25)), 20.0)
        print("[PASS] TEST 19: Descriptive statistics calculations verified mathematically accurate")

    def test_21_calculation_engine_basic_stats(self):
        """TEST 21: Calculation Lab engine basic statistics calculations."""
        res_mean = perform_calculation(self.df_base, category="Basic Statistics", operation="Mean", primary_col="income")
        self.assertAlmostEqual(res_mean["result_value"], 95000.0)

        res_median = perform_calculation(self.df_base, category="Basic Statistics", operation="Median", primary_col="age")
        self.assertAlmostEqual(res_median["result_value"], 47.5)

        res_sum = perform_calculation(self.df_base, category="Basic Statistics", operation="Sum", primary_col="score")
        self.assertAlmostEqual(res_sum["result_value"], 857.8)
        print("[PASS] TEST 21: Calculation Lab engine accurately computed basic statistics")

    def test_22_calculation_engine_relationships(self):
        """TEST 22: Calculation Lab engine relationships and grouped statistics."""
        res_corr = perform_calculation(self.df_base, category="Relationships", operation="Pearson Correlation", primary_col="age", secondary_col="income")
        self.assertAlmostEqual(res_corr["result_value"], 1.0)

        res_grp = perform_calculation(self.df_base, category="Relationships", operation="Grouped Statistics", primary_col="category", secondary_col="score")
        self.assertIn("grouped_df", res_grp)
        self.assertEqual(len(res_grp["grouped_df"]), 2)
        print("[PASS] TEST 22: Calculation Lab engine accurately computed correlation and grouped statistics")

    def test_23_calculation_engine_dataset_comparison(self):
        """TEST 23: Calculation Lab engine dataset comparison delta calculations."""
        df_updated = self.df_base.copy()
        df_updated["income"] = df_updated["income"] * 1.5
        res_cmp = perform_calculation(df_updated, df_prev=self.df_base, category="Dataset Comparison", operation="Mean Change", primary_col="income")
        self.assertAlmostEqual(res_cmp["result_value"], 47500.0)
        print("[PASS] TEST 23: Calculation Lab engine accurately computed dataset comparison deltas")

    def test_24_graph_guide_generator(self):
        """TEST 24: Graph Guide metadata generator for visualizations."""
        guide = get_graph_guide("3D Scatter", "age", "income", "score", "category")
        self.assertIn("3D Scatter", guide["title"])
        self.assertIn("age", guide["x_axis"])
        self.assertIn("category", guide["color_meaning"])
        print("[PASS] TEST 24: Graph Guide metadata generator returned complete descriptive guide")

    def test_25_zero_code_display_check(self):
        """TEST 25: Verify zero st.code or raw code calls exist in user-facing view files."""
        import glob
        view_files = glob.glob("views/*.py")
        code_call_found = False
        for fpath in view_files:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                if "st.code(" in content or "st.echo(" in content:
                    code_call_found = True
                    break
        self.assertFalse(code_call_found, "Raw code calls (st.code/st.echo) were found in a user-facing view file!")
        print("[PASS] TEST 25: Verified ZERO source code displays exist across all user-facing views")

    def test_26_graph_guide_causation_caution(self):
        """TEST 26: Graph Guide causation warning check for correlation charts."""
        g_scatter = get_graph_guide("Scatter", "income", "score")
        self.assertIn("Correlation does not prove causation.", g_scatter["data_science_insight"])

        g_heat = get_graph_guide("Heatmap", "attr1", "attr2")
        self.assertIn("Correlation does not prove causation.", g_heat["data_science_insight"])
        print("[PASS] TEST 26: Verified Graph Guide includes explicit correlation causation warning")

    def test_27_no_st_json_in_views(self):
        """TEST 27: Verify zero st.json calls exist in user-facing components and views."""
        import glob
        files_to_check = glob.glob("views/*.py") + glob.glob("components/*.py")
        json_call_found = False
        for fpath in files_to_check:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                if "st.json(" in content:
                    json_call_found = True
                    break
        self.assertFalse(json_call_found, "st.json() was found in a user-facing component or view!")
        print("[PASS] TEST 27: Verified ZERO st.json raw object outputs exist in user interface")

    def test_28_unique_widget_keys_check(self):
        """TEST 28: Audit widget key uniqueness across all view files."""
        import glob
        import re
        view_files = glob.glob("views/*.py")
        all_keys = []
        duplicate_keys = []
        key_pattern = re.compile(r'key=["\']([^"\']+)["\']')

        for fpath in view_files:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                keys = key_pattern.findall(content)
                for k in keys:
                    if k in all_keys and k not in duplicate_keys:
                        duplicate_keys.append(k)
                    all_keys.append(k)

        self.assertEqual(len(duplicate_keys), 0, f"Duplicate widget keys detected across views: {duplicate_keys}")
        print("[PASS] TEST 28: Verified 100% unique Streamlit widget keys across all view files")

    def test_29_zero_raw_html_in_guides(self):
        """TEST 29: Verify ZERO raw HTML tags (<div, <strong, style=) exist in graph guide dictionary strings."""
        chart_types = ["3D Scatter", "3D Surface", "3D Mesh", "Histogram", "Boxplot", "Heatmap", "Scatter", "Bar Chart"]
        html_found = False
        for ctype in chart_types:
            g = get_graph_guide(ctype, "attr_x", "attr_y", "attr_z", "color_attr")
            for key, val in g.items():
                if isinstance(val, str):
                    if "<div" in val or "<strong" in val or "style=" in val or "</span>" in val:
                        html_found = True
                        break
        self.assertFalse(html_found, "Raw HTML tags were found inside get_graph_guide dictionary strings!")
        print("[PASS] TEST 29: Verified ZERO raw HTML string tags exist in graph guide metadata")

    def test_30_graph_guide_educational_keys(self):
        """TEST 30: Verify Graph Guide dictionaries contain all required educational fields."""
        g_hist = get_graph_guide("Histogram", "age", "frequency")
        required_keys = ["title", "what_it_represents", "x_axis", "y_axis", "what_to_look_for", "how_to_interpret", "data_science_insight"]
        for rk in required_keys:
            self.assertIn(rk, g_hist, f"Missing required educational key '{rk}' in Histogram guide!")
        print("[PASS] TEST 30: Verified Graph Guide contains complete educational fields")

    def test_31_prepare_safe_dataframe_duplicate_columns(self):
        """TEST 31: Duplicate column names position-safe handling without modifying original dataset."""
        df_dup = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["Working Days / Week", "Working Days / Week", "Salary"])
        safe_df, options, option_map, has_dups = prepare_safe_dataframe(df_dup)

        self.assertTrue(has_dups)
        self.assertEqual(len(safe_df.columns), 3)
        self.assertNotEqual(safe_df.columns[0], safe_df.columns[1])
        # Ensure original df remains untouched
        self.assertEqual(df_dup.columns[0], df_dup.columns[1])
        print("[PASS] TEST 31: Duplicate column names safely disambiguated without mutating original dataset")

    def test_32_pre_plot_axis_validation(self):
        """TEST 32: Pre-plot X/Y/Z validation prevents invalid Plotly calls and tracebacks."""
        df_test = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": ["cat1", "cat2", "cat3"]})
        valid_3d, msg_3d = validate_chart_inputs(df_test, "A", "B", "C", "3D Scatter")
        self.assertFalse(valid_3d)
        self.assertIn("non-numeric", msg_3d)

        valid_same, msg_same = validate_chart_inputs(df_test, "A", "A", "B", "3D Scatter")
        self.assertFalse(valid_same)
        self.assertIn("distinct attributes", msg_same)
        print("[PASS] TEST 32: Pre-plot validation safely caught non-numeric and identical 3D axis constraints")

    def test_33_pipeline_card_components(self):
        """TEST 33: Verify Calculation -> Insight Chain pipeline card components exist."""
        from components import cards
        self.assertTrue(hasattr(cards, "pipeline_card"))
        self.assertTrue(hasattr(cards, "data_story_card"))
        self.assertTrue(hasattr(cards, "analyst_readiness_card"))
        self.assertTrue(hasattr(cards, "next_best_analysis_card"))
        print("[PASS] TEST 33: Verified all signature pipeline and analytical readiness cards exist")

    def test_34_3d_visual_modeling_quality(self):
        """TEST 34: 3D Surface and Mesh generation with valid numeric datasets."""
        df_3d = pd.DataFrame({"x": np.random.rand(50), "y": np.random.rand(50), "z": np.random.rand(50)})
        fig_surface, meta_s = create_3d_surface(df_3d, "x", "y", "z")
        fig_mesh, meta_m = create_3d_mesh(df_3d, "x", "y", "z")
        self.assertIsNotNone(fig_surface)
        self.assertIsNotNone(fig_mesh)
        self.assertEqual(meta_s["valid_rows"], 50)
        print("[PASS] TEST 34: 3D Surface and 3D Mesh modeling generated cleanly from numeric dataset")

    def test_35_design_system_css_build(self):
        """TEST 35: Verify design system _build_css generates valid CSS without f-string NameErrors."""
        from components.design_system import get_tokens, _build_css
        tokens = get_tokens("dark")
        css_str = _build_css(tokens)
        self.assertIn(".stTabs", css_str)
        self.assertGreater(len(css_str), 5000)
        print("[PASS] TEST 35: Verified design system CSS builds cleanly without f-string NameErrors")

    def test_36_zero_connection_badges_in_header_and_sidebar(self):
        """TEST 36: Verify zero developer connection badges (Gemini Connected/ML engine available) exist in app UI."""
        with open("components/layout.py", "r", encoding="utf-8") as f:
            layout_content = f.read()
        self.assertNotIn("Gemini Connected", layout_content)

        with open("app.py", "r", encoding="utf-8") as f:
            app_content = f.read()
        self.assertNotIn("Gemini AI connected", app_content)
        self.assertNotIn("ML engine available", app_content)
        self.assertIn("SYSTEM CAPABILITIES", app_content)
        print("[PASS] TEST 36: Verified zero backend diagnostic badges in UI and System Capabilities rail active")

    def test_37_executive_report_heading_formatting(self):
        """TEST 37: Verify Executive AI Report heading formatters for Web UI and downloads."""
        from ui import format_report_headings_for_web
        from utils import format_report_for_download

        sample_report = "# Executive Assessment\nThe dataset is highly complete.\n\n# Key Findings\nData quality score 92/100."
        web_html = format_report_headings_for_web(sample_report)
        download_txt = format_report_for_download(sample_report)

        self.assertIn("EXECUTIVE ASSESSMENT", web_html)
        self.assertIn("border-left: 4px solid #6366f1", web_html)
        self.assertIn("============================================================", download_txt)
        print("[PASS] TEST 37: Verified Executive AI report heading callouts and download formatters operate cleanly")

    def test_38_zero_ml_rendering_artifacts(self):
        """TEST 38: Verify zero raw rendering artifacts ([svg], localhost:8501, canvascanvas) in ML view."""
        with open("views/ml_view.py", "r", encoding="utf-8") as f:
            ml_code = f.read()

        self.assertNotIn("[svg]", ml_code)
        self.assertNotIn("localhost:8501", ml_code)
        self.assertNotIn("canvascanvas", ml_code)
        self.assertIn("_clean_text_artifacts", ml_code)
        self.assertIn("Your original dataset is protected. ML analysis is performed on an in-memory copy, so the uploaded file remains unchanged.", ml_code)
        print("[PASS] TEST 38: Verified ZERO raw SVG/canvas/localhost rendering artifacts in ML workspace view")



if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AI DATASET EXPLORER MANDATORY TEST SUITE")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIDatasetIntelligence)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nALL TEST CASES PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)






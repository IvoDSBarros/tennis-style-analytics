"""
ai_narrative_generator.py - AI NARRATIVE GENERATOR
----------------------
TENNIS PLAYING STYLE ANALYSIS
@AUTHOR: Ivo Barros
----------------------
GOAL: Generate AI-powered tennis playing style narratives for PCA visualizations.
Outputs 2 sections per surface: Scores (scatter plot explanation) 
and Loadings (variable interpretation).

INPUT:
    data/pca_scores_by_surface.json
    data/pca_loadings_by_surface.json
    data/pca_variance_by_surface.json
    data/pca_clusters.json

OUTPUT:
    data/pca_crew_ai_narrative.json - Consolidated narratives for all surfaces

USAGE:
    python ai_narrative/ai_narrative_generator.py
"""

import os
import sys
import json
import time
import re
from typing import List, Type, Dict
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

#------------------------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------------------------
load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_STRING = "gemini/gemini-2.5-flash"

if not GEMINI_API_KEY:
    print("FATAL ERROR: GEMINI_API_KEY is not set in environment or .env file.")
    sys.exit(1)

#------------------------------------------------------------------------------
# PATHS
#------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.absolute()
PATH_DATA = ROOT_DIR / "data"

# Input files
SCORES_TABLE_PATH = PATH_DATA / "pca_scores_by_surface.json"
LOADINGS_TABLE_PATH = PATH_DATA / "pca_loadings_by_surface.json"
VARIANCE_TABLE_PATH = PATH_DATA / "pca_variance_by_surface.json"
CLUSTERS_TABLE_PATH = PATH_DATA / "pca_clusters.json"

# Output file (consolidated)
OUTPUT_NARRATIVE_PATH = PATH_DATA / "pca_crew_ai_narrative.json"

print("=" * 60)
print("AI NARRATIVE GENERATOR (Step 9)")
print("=" * 60)
print(f"Input scores: {SCORES_TABLE_PATH}")
print(f"Input loadings: {LOADINGS_TABLE_PATH}")
print(f"Input variance: {VARIANCE_TABLE_PATH}")
print(f"Input clusters: {CLUSTERS_TABLE_PATH}")
print(f"Output: {OUTPUT_NARRATIVE_PATH}")
print("-" * 60)

# Load data
with open(SCORES_TABLE_PATH, 'r') as file:
    data_scores = json.load(file)

with open(LOADINGS_TABLE_PATH, 'r') as file:
    data_loadings = json.load(file)

with open(VARIANCE_TABLE_PATH, 'r') as file:
    data_variance = json.load(file)

with open(CLUSTERS_TABLE_PATH, 'r') as file:
    data_clusters = json.load(file)

#------------------------------------------------------------------------------
# CLEAN THINKING SECTIONS
#------------------------------------------------------------------------------
def clean_thinking_sections(text: str, surface: str) -> str:
    """Remove all thinking/planning sections from output"""
    
    split_markers = [
        "I have successfully retrieved",
        "I will now",
        "Let me",
        "Scores Section Plan:",
        "Loadings Section Plan:",
        "I will ensure to follow",
        "Here is the analysis",
        f"# {surface} Surface: PCA Analysis"
    ]
    
    for marker in split_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                if marker.startswith("#"):
                    text = marker + parts[1]
                else:
                    text = parts[1].strip()
                break
    
    expected_header = f"# {surface} Surface: PCA Analysis"
    if not text.startswith(expected_header):
        pos = text.find(expected_header)
        if pos != -1:
            text = text[pos:]
        else:
            text = expected_header + "\n\n" + text.lstrip('# ')
    
    # Remove any remaining thinking patterns
    lines = text.split('\n')
    cleaned_lines = []
    skip = False
    
    for line in lines:
        if any(phrase in line.lower() for phrase in ['i have', 'i will', 'let me', 'plan:', 'step ', 'first, i']):
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

#------------------------------------------------------------------------------
# TOOL DEFINITION
#------------------------------------------------------------------------------
class TennisDataEnforcerSchema(BaseModel):
    surface_name: str = Field(description="Tennis surface: Clay, Grass, or Hard")

class TennisDataEnforcer(BaseTool):
    name: str = "TennisDataEnforcer"
    description: str = "Provides clean, token-optimized PCA scores, loadings, and variance data with pre-calculated mathematical extremes."
    args_schema: Type[BaseModel] = TennisDataEnforcerSchema

    def _run(self, surface_name: str) -> str:
        try:
            print(f"\n  Retrieving PCA data for {surface_name}...")
            
            # 1. Clean scores: Strip redundant surface strings and round floats
            scores = [
                {
                    "player": item["player"], 
                    "group": item["group"], 
                    "PC1": round(item["PC1"], 4), 
                    "PC2": round(item["PC2"], 4)
                }
                for item in data_scores if item.get('surface') == surface_name
            ]
            
            # 2. Clean loadings: Strip redundant surface strings and round floats
            loadings = [
                {
                    "variable": item["variable"], 
                    "PC1": round(item["PC1"], 4), 
                    "PC2": round(item["PC2"], 4)
                }
                for item in data_loadings if item.get('surface') == surface_name
            ]
            
            # 3. Clean variance: Convert to clean percentages
            variance = [
                {
                    "component": item["component"], 
                    "explained_variance_pct": round(item["explained_variance_ratio"] * 100, 1),
                    "cumulative_variance_pct": round(item["cumulative_variance_ratio"] * 100, 1)
                }
                for item in data_variance if item.get('surface') == surface_name
            ]
            
            if not scores or not loadings or not variance:
                return json.dumps({"error": f"Surface '{surface_name}' not found in data"}, indent=2)
            
            # 4. Pre-calculate extreme boundaries (mathematical guardrails)
            extremes = {
                "pc1_highest": {"player": max(scores, key=lambda x: x["PC1"])["player"], "value": max(scores, key=lambda x: x["PC1"])["PC1"]},
                "pc1_lowest": {"player": min(scores, key=lambda x: x["PC1"])["player"], "value": min(scores, key=lambda x: x["PC1"])["PC1"]},
                "pc2_highest": {"player": max(scores, key=lambda x: x["PC2"])["player"], "value": max(scores, key=lambda x: x["PC2"])["PC2"]},
                "pc2_lowest": {"player": min(scores, key=lambda x: x["PC2"])["player"], "value": min(scores, key=lambda x: x["PC2"])["PC2"]}
            }
            
            # 5. Pre-calculate negative loadings (prevent LLM from missing them)
            negative_loadings = {
                "pc1_negative": [{"variable": l["variable"], "loading": l["PC1"]} for l in loadings if l["PC1"] < 0],
                "pc2_negative": [{"variable": l["variable"], "loading": l["PC2"]} for l in loadings if l["PC2"] < 0]
            }
            
            # 6. Clusters          
            clusters = data_clusters['visualization']
            clusters_clusters = clusters[surface_name]['clusters']
            clusters_points = clusters[surface_name]['points']
            clusters_n_clusters = len(clusters_clusters)
                        
            # 7. Package everything using flat structure
            result = {
                "surface": surface_name,
                "variance": variance,
                "pre_calculated_extremes": extremes,
                "pre_calculated_negative_loadings": negative_loadings,
                "scores": scores,
                "loadings": loadings,
                "clusters": clusters_clusters,
                "clusters_points": clusters_points,
                "clusters_n_clusters": clusters_n_clusters
            }
            
            print(f"    Retrieved {len(scores)} player records")
            print(f"    Retrieved {len(loadings)} variable loadings")
            print(f"    PC1 extremes: {extremes['pc1_highest']['player']} ({extremes['pc1_highest']['value']}) / {extremes['pc1_lowest']['player']} ({extremes['pc1_lowest']['value']})")
            print(f"    Negative PC1 loadings found: {len(negative_loadings['pc1_negative'])}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)

#------------------------------------------------------------------------------
# CREW DEFINITION
#------------------------------------------------------------------------------
def create_crew():
    tennis_enforcer = TennisDataEnforcer()
    
    analyst = Agent(
        role="Tennis PCA Analyst",
        goal="Explain PCA results for tennis playing styles with mathematical accuracy",
        backstory="Expert at interpreting PCA components. You always check negative loadings and pre-calculated extremes before making claims.",
        tools=[tennis_enforcer],
        allow_delegation=False,
        verbose=False,
        llm=GEMINI_MODEL_STRING,
        max_rpm=8
    )
    
    analysis_task = Task(
        description=(
            "Analyze PCA data for {surface_name} surface.\n\n"
            
            "DATA RETRIEVAL:\n"
            "Use TennisDataEnforcer to get scores, loadings, variance, clusters_clusters, clusters_points, and clusters_n_clusters.\n"
            "The tool provides pre_calculated_extremes and pre_calculated_negative_loadings - use these for accurate math.\n\n"
            
            "ROLE & CORE OBJECTIVE:\n"
            "You are an elite, senior tennis analyst writing data-driven narrative profiles for a premium sports analytics platform. "
            "Your goal is to translate Principal Component Analysis (PCA) biplot data into deep, contextual on-court strategic realities for the selected surface without ever breaking real-world tennis mechanics.\n\n"
            
            "MANDATORY PRE-WRITING PROTOCOL (EXECUTE BEFORE WRITING):\n"
            "Step 1: Extract the loadings data for this specific surface from TennisDataEnforcer.\n"
            "Step 2: For each metric, identify its PRIMARY axis (the axis where its absolute loading is highest).\n"
            "Step 3: For each metric, check its DIRECTIONAL SIGN on each axis. A positive loading pushes toward the positive end. A negative loading pulls toward the negative end.\n"
            "Step 4: For PC1, separate metrics into POSITIVE PUSHERS and NEGATIVE PULLERS. List them separately.\n"
            "Step 5: For PC2, repeat Step 4.\n"
            "Step 6: Before writing about any player, verify that you are describing metrics on the correct axis with the correct directional sign.\n\n"
            
            "DATA-TO-TENNIS TRUTH GUARDRAILS (ANTI-HALLUCINATION):\n"
            "- THE ONE-SIDED AXIS RULE: If an axis contains exclusively one-sided metric weights (all variables load positively OR all load negatively), you must explicitly state that there are zero opposing metrics pulling in the opposite direction. Never fabricate a competing tactical style. Characterize a low score on a one-sided axis strictly as a structural deficit or absence of those traits.\n"
            "- THE PRIMARY-SECONDARY LOADING RULE: Metrics do not 'belong' exclusively to one axis. A metric may load strongly on PC1 and weakly on PC2, or vice versa. When describing a player's position: use the PRIMARY axis (highest absolute loading) to explain the metric's dominant influence. Acknowledge a SECONDARY influence ONLY if the loading exceeds 0.3 on that axis. If a metric loads below 0.3 on an axis, it has NEGLIGIBLE influence and must be ignored for that dimension.\n"
            "- THE LOADING MAGNITUDE RULE: Before attributing any tactical meaning to a player's position on an axis, identify which metrics have the absolute highest loading magnitudes on THAT axis. Name the metric with the SINGLE HIGHEST absolute loading first when describing what the axis represents. Secondary metrics (loading 0.3-0.6) add nuance but should not replace the dominant metric.\n"
            "- THE COORDINATE-TO-METRIC INTERPRETATION PROTOCOL: A player's (PC1, PC2) coordinate results from the combined influence of ALL metrics projecting onto BOTH axes. When interpreting a position: Step A: For PC1, identify the metrics with the LARGEST ABSOLUTE LOADINGS. These exert the strongest pull on horizontal position. Use the SIGN of each loading to determine whether a positive or negative PC1 score indicates higher or lower values of that metric. Step B: For PC2, repeat Step A using PC2 loadings. Step C: A metric may load on BOTH axes. If Drop Shot Effectiveness loads at 0.62 on PC1 and -0.20 on PC2, it primarily drives PC1 (0.62 > 0.20) but also contributes a small negative influence to PC2. NEVER state that a metric 'belongs' exclusively to one axis. Instead, describe its PRIMARY influence and acknowledge any SECONDARY influence if meaningful (absolute loading > 0.3). Step D: When describing a player's tactical profile, prioritize metrics that load strongly (absolute loading > 0.5) on their respective axes.\n"
            "- THE VARIANCE EXPLANATION RULE: Never blindly equate a low score on an efficiency axis with a passive or defensive style. Distinguish between low-efficiency defensive play (low winners, low errors) and low-efficiency high-risk play (high winners, high errors).\n"
            "- SURFACE CONTEXT RULE: On Clay, high-touch metrics represent patient point construction. On Grass, those same metrics represent high-risk, creative gambles meant to shorten rallies.\n"
            "- THE DIRECTIONAL SIGN VERIFICATION RULE: Before describing any metric's influence on an axis, you MUST check its SIGN. A metric with a NEGATIVE loading pulls players toward the NEGATIVE end of that axis. A metric with a POSITIVE loading pushes players toward the POSITIVE end. When describing an axis, explicitly separate metrics by their directional sign: 'Metric A pushes toward the positive end, while Metric B and Metric C pull in the opposite direction.' Never describe a negative-loading metric as if it contributes positively to that axis.\n\n"
            
            "TONE & VOICE GUARDRAILS (STRICTLY ENFORCED):\n"
            "- NO SELF-REFERENTIAL LANGUAGE: Absolutely ban phrases like 'our analysis,' 'the data shows,' 'this study reveals,' 'the chart indicates,' or 'as seen in the dataset.' Write as a human analyst.\n"
            "- THE TENNIS ACCENT RULE: Speak directly about players and their physical actions. Write 'A fiercely defensive camp emerges on the baseline' NOT 'Our cluster analysis identifies a defensive camp.'\n"
            "- BANNED JARGON: Never use 'components,' 'data points,' 'taxonomic clusters,' 'statistically grouped,' 'variables,' or 'snake_case_names.'\n"
            "- DYNAMIC PROSE: Describe groupings naturally: 'the return-oriented contingent,' 'the heavy-serving cohort,' 'this tactical circle.'\n\n"
            
            "STRICT FORMATTING & VARIABLE MAPPING:\n"
            "- PERCENTAGE FORMATTING: Every percentage value MUST be formatted as XX.X% (e.g., '63.3%', '38.0%', never '63%', '38.21%').\n"
            "- VARIABLE DISPLAY MAPPING (STRICT TRANSLATION - NEVER USE RAW TERMS):\n"
            "    winner_error_ratio → Winner/UE Ratio\n"
            "    drop_shot_effectiveness → Drop Shot Effectiveness\n"
            "    break_point_conversion_pct → Break Point Conversion %\n"
            "    net_points_won_pct → Net Points Won %\n"
            "    first_serve_points_won or first_serve_pts_won → 1st Serve Points Won %\n"
            "    second_serve_points_won_pct → 2nd Serve Points Won %\n"
            "    ace_pct → Ace %\n"
            "    double_fault_pct → Double Fault %\n"
            "    break_points_saved_pct → Break Points Saved %\n"
            "    unreturned_serve_rate → Unreturned Serve Rate\n"
            "    first_serve_pct → 1st Serve %\n\n"
            
            "OUTPUT STRUCTURE & LENGTH GUARDRAILS:\n"
            "Your output must match this exact structural blueprint.\n\n"
            
            "# {surface_name} Surface: PCA Analysis\n\n"
            "## Scores\n"
            "[Deliver exactly 4 paragraphs. Each paragraph MUST be strictly between 145-155 words. Absolutely no bullet points, lists, or bold inline headers.]\n"
            "Paragraph 1 - THE TACTICAL SPECTRUM: Translate the axes into on-court behavior using loadings data. State exact variance percentages to one decimal place. Identify the four extreme boundary players (max/min on both axes). Do NOT mention Cluster 0, Cluster 1, etc. When describing what defines an axis, you MUST explicitly separate metrics by their directional sign. For example: 'Metric A pushes players toward the positive end of PC1, while Metric B and Metric C pull in the opposite direction.' Never group metrics of opposite signs together as if they all push in the same direction.\n"
            "Paragraph 2 - THE NEW GENERATION CONTRAST (ALCARAZ VS SINNER): CRITICAL: Contrast Sinner and Alcaraz against each other BEFORE comparing to Big 3. State their exact (PC1, PC2) coordinates. Apply the COORDINATE-TO-METRIC INTERPRETATION PROTOCOL: identify which metrics exert the strongest pull on their PC1 position using PC1 loadings. Identify which metrics exert the strongest pull on their PC2 position using PC2 loadings. If a metric has a primary influence on PC1, use it to explain their PC1 position. If a metric has a secondary influence on PC2, acknowledge it only if loading > 0.3. Frame their profiles as execution within this specific 8-match sample, not career generalizations.\n"
            "Paragraph 3 - COMPARISON TO THE BIG 3: State exact coordinates for Federer, Nadal, and Djokovic. Apply COORDINATE-TO-METRIC INTERPRETATION PROTOCOL to evaluate whether Sinner and Alcaraz match a Big 3 blueprint or carve out unique space.\n"
            "Paragraph 4 - TACTICAL CAMPS & STRATEGIC GRAVITY:\n"
                "- Use 'clusters_clusters' for complete cluster membership and 'clusters_points' for positioning. 'clusters_n_clusters' gives the total count.\n"
                "- CRITICAL - CHECK CLUSTER SPREAD BEFORE LABELING: Examine the full range of coordinates within each cluster before assigning a tactical label. If PC1 or PC2 spans more than 2 units, you MUST describe it as a 'mixed' or 'broad' tactical zone. Only assign a specific label (e.g., 'serve-centric,' 'counter-punching') if the majority of players in that cluster share that trait. This check overrides all other characterization rules.\n"
                "- Orient each cluster by precise visual geography (e.g., 'upper-right quadrant,' 'lower-left quadrant'). Never use cluster numbers.\n"
                "- Anchor with 2-3 prominent players who best exemplify that tactical identity. Do NOT enumerate every player—let tactical description imply membership.\n"
                "- Describe what the cluster centroid represents on court using dominant loadings for that region.\n"
                "- Place Sinner and Alcaraz into their respective clusters using metric alignment.\n"
                "- Address any extreme outliers on the periphery and note whether 'clusters_n_clusters' indicates a single dominant group or multiple distinct tactical camps.\n\n"

            "## Loadings\n"
            "[Deliver exactly 4 paragraphs. Each paragraph MUST be strictly between 145-155 words. Absolutely no bullet points, lists, or bold inline headers.]\n"
            "Paragraph 1 - PC1 METRIC ANALYSIS: Identify which metrics push players toward positive PC1. Explicitly call out negative pull vectors. Apply the ONE-SIDED AXIS RULE if applicable. Name the DOMINANT metric first. State the exact explained variance percentage for PC1.\n"
            "Paragraph 2 - PC1 TACTICAL STYLE: Explain the playing style PC1 represents based on the strategic tension or structural deficit. Apply SURFACE CONTEXT RULE. Describe what a player at the positive extreme looks like on court versus a player at the negative extreme.\n"
            "Paragraph 3 - PC2 METRIC ANALYSIS: Identify metrics carrying the strongest positive weights on PC2. Contrast them directly against metrics carrying negative weights. Apply the PRIMARY-SECONDARY LOADING RULE. State the exact explained variance percentage for PC2.\n"
            "Paragraph 4 - AXES INTERPLAY: Explain how both axes work together. Frame it so the reader understands how two players at the same vertical height can have opposing tactical identities based on horizontal separation. Describe how the dimensions combine to map out diverse playing styles.\n\n"
            
            "FORMAT:\n"
            "# {surface_name} Surface: PCA Analysis\n\n"
            "## Scores\n"
            "[4 paragraphs]\n\n"
            "## Loadings\n"
            "[4 paragraphs]\n"
        ),
        expected_output="2 sections: Scores with individual Sinner/Alcaraz contrast, Loadings with negative loadings explicitly identified",
        agent=analyst,
        name="Tennis PCA Analysis",
        markdown=True
    )
    
    return Crew(agents=[analyst], tasks=[analysis_task], process=Process.sequential, verbose=False)

#------------------------------------------------------------------------------
# MAIN - GENERATE NARRATIVES FOR ALL SURFACES
#------------------------------------------------------------------------------
def main():
    valid_surfaces = ['Clay', 'Grass', 'Hard']
    
    # Consolidated narratives for all surfaces
    all_narratives = {}
    timestamp = int(time.time())
    
    for surface in valid_surfaces:
        print(f"\n{'='*40}")
        print(f"Generating narrative for {surface} Surface")
        print(f"{'='*40}")
        
        success = False
        retries = 0
        max_retries = 3
        
        while not success and retries <= max_retries:
            try:
                if retries > 0:
                    print(f"  Retry {retries}/{max_retries} for {surface}...")
                
                # Create and run crew
                crew = create_crew()
                result = crew.kickoff(inputs={"surface_name": surface})
                
                # Clean the thinking sections
                raw_text = str(result)
                cleaned_report = clean_thinking_sections(raw_text, surface)
                
                # Extract narrative sections
                scores_narrative = ""
                loadings_narrative = ""
                
                if "## Scores" in cleaned_report and "## Loadings" in cleaned_report:
                    scores_part = cleaned_report.split("## Scores")[1].split("## Loadings")[0].strip()
                    loadings_part = cleaned_report.split("## Loadings")[1].strip()
                    
                    if "##" in loadings_part:
                        loadings_part = loadings_part.split("##")[0].strip()
                    
                    scores_narrative = scores_part
                    loadings_narrative = loadings_part
                
                # Store in consolidated dictionary
                all_narratives[surface] = {
                    "scores_narrative": scores_narrative,
                    "loadings_narrative": loadings_narrative,
                    "status": "success"
                }
                
                print(f"  ✓ Narrative generated for {surface}")
                print(f"    Scores: {len(scores_narrative)} chars")
                print(f"    Loadings: {len(loadings_narrative)} chars")
                success = True
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Error: {error_msg[:200]}...")
                
                # Extract retry delay from error message
                retry_delay = 15  # Default fallback
                match = re.search(r'retry in ([0-9.]+)s', error_msg)
                if match:
                    retry_delay = float(match.group(1)) + 3
                    print(f"  ⏳ Quota limit. Waiting {retry_delay:.1f}s before retry...")
                elif "503" in error_msg or "UNAVAILABLE" in error_msg:
                    retry_delay = 30
                    print(f"  ⏳ Server overload. Waiting {retry_delay}s before retry...")
                else:
                    retry_delay = 15
                    print(f"  ⏳ Waiting {retry_delay}s before retry...")
                
                retries += 1
                
                if retries > max_retries:
                    print(f"  ✗ Failed after {max_retries} retries. Skipping {surface}.")
                    all_narratives[surface] = {
                        "scores_narrative": "",
                        "loadings_narrative": "",
                        "status": f"failed: {error_msg[:100]}"
                    }
                else:
                    time.sleep(retry_delay)
    
    # Save consolidated JSON for all surfaces
    consolidated_output = {
        "generated_timestamp": timestamp,
        "model": GEMINI_MODEL_STRING,
        "surface": all_narratives
    }
    
    with open(OUTPUT_NARRATIVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(consolidated_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*40}")
    print("Saving Results")
    print(f"{'='*40}")
    print(f"  Saved: {OUTPUT_NARRATIVE_PATH}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("NARRATIVE GENERATION COMPLETE")
    print("=" * 60)
    for surface in valid_surfaces:
        if surface in all_narratives:
            status = all_narratives[surface].get("status", "unknown")
            if status == "success":
                scores_len = len(all_narratives[surface].get("scores_narrative", ""))
                loadings_len = len(all_narratives[surface].get("loadings_narrative", ""))
                print(f"  {surface}: ✓ {scores_len + loadings_len} chars total")
            else:
                print(f"  {surface}: ✗ {status}")
    
    print(f"\n✓ Results saved to: {OUTPUT_NARRATIVE_PATH}")

if __name__ == '__main__':
    main()
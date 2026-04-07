
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


class TravelPlannerState(TypedDict):
    user_input: str        # Raw text from the user
    destination: str       # Extracted destination city/country
    duration: str          # Trip duration (e.g. "7 days")
    travel_style: str      # e.g. adventure, cultural, relaxation
    budget_range: str      # low / medium / high
    destination_info: str  # Research output from Agent 2
    itinerary: str         # Day-by-day plan from Agent 3
    budget_estimate: str   # Cost breakdown from Agent 4
    final_plan: str        # Assembled output



def get_llm(temperature: float = 0.7) -> ChatGroq:
    """Return a Groq LLM instance. Exits with a helpful message if key is missing."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable is not set.")
        print("  Get a FREE key at: https://console.groq.com/keys")
        print("  Then run:  export GROQ_API_KEY='gsk_...'")
        sys.exit(1)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=api_key
    )


def input_analyzer_agent(state: TravelPlannerState) -> TravelPlannerState:


    print("\n🔍 [Agent 1 — Input Analyzer] Processing your travel request...")

    llm = get_llm(temperature=0.2)   # low temp for structured extraction

    system_prompt = """You are a Travel Input Analyzer AI.

    Your job is to extract four pieces of information from any free-text travel request:
    1. DESTINATION  – the city or country the user wants to visit
    2. DURATION     – how many days (default to 7 if not mentioned)
    3. TRAVEL_STYLE – one of: adventure | cultural | relaxation | food | luxury | budget
    4. BUDGET_RANGE – one of: low | medium | high (guess from context if not stated)

    Respond ONLY in this exact format (no extra lines):
    DESTINATION: <value>
    DURATION: <value>
    TRAVEL_STYLE: <value>
    BUDGET_RANGE: <value>
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Analyze this travel request:\n\n{state['user_input']}")
    ]

    response = llm.invoke(messages)
    text = response.content.strip()

    # Parse the key: value lines into a dict
    parsed: dict = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip().upper()] = value.strip()

    state["destination"]   = parsed.get("DESTINATION",   "Unknown Destination")
    state["duration"]      = parsed.get("DURATION",      "7 days")
    state["travel_style"]  = parsed.get("TRAVEL_STYLE",  "cultural")
    state["budget_range"]  = parsed.get("BUDGET_RANGE",  "medium")

    print(f"   ✅  Destination  : {state['destination']}")
    print(f"   ✅  Duration     : {state['duration']}")
    print(f"   ✅  Travel Style : {state['travel_style']}")
    print(f"   ✅  Budget Range : {state['budget_range']}")

    return state


def destination_research_agent(state: TravelPlannerState) -> TravelPlannerState:
    print(f"\n🌍 [Agent 2 — Destination Researcher] Researching {state['destination']}...")

    llm = get_llm(temperature=0.5)

    system_prompt = """You are a world-class Destination Research Expert.

    For the given destination, provide a concise but information-rich overview covering:
    • Top 5 must-see attractions (with 1-sentence description each)
    • Best season to visit and typical weather
    • Local culture, customs, and etiquette tips
    • Getting around (airport transfers, public transport, taxis)
    • Recommended neighbourhoods / areas to stay
    • 3–5 signature dishes or drinks to try

    Keep the total length to roughly 350–450 words.
    Write in clear, engaging prose — no excessive bullet nesting.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Research destination: {state['destination']}\n"
            f"Trip duration      : {state['duration']}\n"
            f"Traveller style    : {state['travel_style']}\n"
        ))
    ]

    response = llm.invoke(messages)
    state["destination_info"] = response.content.strip()

    print(f"   ✅  Research complete ({len(state['destination_info'].split())} words)")
    return state


def itinerary_planner_agent(state: TravelPlannerState) -> TravelPlannerState:
    print(f"\n📅 [Agent 3 — Itinerary Planner] Building your {state['duration']} schedule...")

    llm = get_llm(temperature=0.8) 

    system_prompt = """You are an expert Travel Itinerary Planner.

    Using the destination research provided, create a detailed day-by-day itinerary.

    Rules:
    • Label each day clearly: "Day 1 — <theme>"
    • Split each day into Morning / Afternoon / Evening
    • Include specific place names, not vague descriptions
    • Recommend one restaurant per evening with a reason to visit it
    • Adapt activities to the stated travel style and budget level
    • Add one practical travel tip per day (e.g. best time to arrive, how to book tickets)
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Destination  : {state['destination']}\n"
            f"Duration     : {state['duration']}\n"
            f"Travel style : {state['travel_style']}\n"
            f"Budget level : {state['budget_range']}\n\n"
            f"--- Destination Research ---\n{state['destination_info']}"
        ))
    ]

    response = llm.invoke(messages)
    state["itinerary"] = response.content.strip()

    print(f"   ✅  Itinerary complete ({state['duration']} planned)")
    return state

def budget_estimator_agent(state: TravelPlannerState) -> TravelPlannerState:

    print(f"\n💰 [Agent 4 — Budget Estimator] Calculating costs for {state['destination']}...")

    llm = get_llm(temperature=0.3)   # low temp for factual figures

    system_prompt = """You are a Travel Budget Expert.

    Produce a realistic cost breakdown for the described trip.
    Present three tiers — Budget / Mid-range / Luxury — for each category.

    Categories to cover:
    1. Flights (round-trip per person)
    2. Accommodation (per night × number of nights)
    3. Meals & dining (per day)
    4. Activities & entrance fees (total)
    5. Local transportation
    6. Shopping & miscellaneous

    Finish with a TOTAL ESTIMATED COST row for each tier.
    Format as a clean Markdown table.
    Add 2–3 money-saving tips at the end.
    """

    # Send only the first 600 chars of the itinerary to keep tokens reasonable
    itinerary_snippet = state["itinerary"][:600] + "..." if len(state["itinerary"]) > 600 else state["itinerary"]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Destination   : {state['destination']}\n"
            f"Duration      : {state['duration']}\n"
            f"Travel style  : {state['travel_style']}\n"
            f"Budget level  : {state['budget_range']}\n\n"
            f"--- Itinerary Snippet ---\n{itinerary_snippet}"
        ))
    ]

    response = llm.invoke(messages)
    state["budget_estimate"] = response.content.strip()

    print(f"   ✅  Budget estimate complete")
    return state


def final_assembler(state: TravelPlannerState) -> TravelPlannerState:

    print("\n✨ [Assembler] Compiling your complete travel plan...")

    separator   = "=" * 62
    thin_line   = "─" * 62

    plan = f"""
    {separator}
        🌟  PERSONALISED TRAVEL PLAN  🌟
        Generated by Multi-Agent AI System
    {separator}

    📍 Destination  : {state['destination']}
    ⏱  Duration     : {state['duration']}
    🎯 Travel Style : {state['travel_style'].capitalize()}
    💵 Budget Range : {state['budget_range'].capitalize()}

    {thin_line}
    🌍  DESTINATION OVERVIEW
    {thin_line}
    {state['destination_info']}

    {thin_line}
    📅  DAY-BY-DAY ITINERARY
    {thin_line}
    {state['itinerary']}

    {thin_line}
    💰  BUDGET BREAKDOWN
    {thin_line}
    {state['budget_estimate']}

    {separator}
    ✈️   Safe travels — have an amazing trip!
    {separator}
    """

    state["final_plan"] = plan

    output_path = "travel_plan_output.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Travel Request: {state['user_input']}\n\n")
            f.write(plan)
        print(f"   ✅  Plan saved to: {output_path}")
    except OSError as e:
        print(f"   ⚠️  Could not save file: {e}")

    return state


def build_graph() -> "CompiledStateGraph":

    workflow = StateGraph(TravelPlannerState)

    # ── Register nodes ──────────────────────────────────
    workflow.add_node("input_analyzer",       input_analyzer_agent)
    workflow.add_node("destination_research", destination_research_agent)
    workflow.add_node("itinerary_planner",    itinerary_planner_agent)
    workflow.add_node("budget_estimator",     budget_estimator_agent)
    workflow.add_node("final_assembler",      final_assembler)

    # ── Define edges (execution order) ──────────────────
    workflow.set_entry_point("input_analyzer")
    workflow.add_edge("input_analyzer",       "destination_research")
    workflow.add_edge("destination_research", "itinerary_planner")
    workflow.add_edge("itinerary_planner",    "budget_estimator")
    workflow.add_edge("budget_estimator",     "final_assembler")
    workflow.add_edge("final_assembler",      END)

    return workflow.compile()



def main():
    print("=" * 62)
    print("   🌐  MULTI-AGENT TRAVEL PLANNER")
    print("   Powered by LangChain + LangGraph + Groq (Llama 3.3 70B)")
    print("=" * 62)
    print()
    print("  Four specialised AI agents will collaborate to build")
    print("  your personalised travel plan:")
    print()
    print("   1. 🔍  Input Analyzer      — understands your request")
    print("   2. 🌍  Destination Researcher — gathers destination info")
    print("   3. 📅  Itinerary Planner   — creates your day-by-day schedule")
    print("   4. 💰  Budget Estimator    — calculates realistic costs")
    print()

    # ── Dynamic user input ───────────────────────────────
    print("Describe your dream trip (press Enter to use the example):")
    print("  Example: 'I want to visit Kyoto, Japan for 6 days.")
    print("            I love culture and food. Medium budget.'")
    print()

    raw = input("Your travel request: ").strip()

    if not raw:
        raw = (
            "I want to visit Kyoto, Japan for 6 days. "
            "I love Japanese culture, temples, and food. Medium budget."
        )
        print(f"\n[Using example request]: {raw}")

    # ── Initialise shared state ──────────────────────────
    initial_state: TravelPlannerState = {
        "user_input":        raw,
        "destination":       "",
        "duration":          "",
        "travel_style":      "",
        "budget_range":      "",
        "destination_info":  "",
        "itinerary":         "",
        "budget_estimate":   "",
        "final_plan":        "",
    }

    # ── Build graph and run ──────────────────────────────
    print("\n" + "─" * 62)
    print("  WORKFLOW STARTED — agents are collaborating...")
    print("─" * 62)

    app = build_graph()
    final_state = app.invoke(initial_state)

    # ── Print the assembled plan ─────────────────────────
    print(final_state["final_plan"])
    print("\n[Done] Your travel plan has also been saved to travel_plan_output.txt")


if __name__ == "__main__":
    main()

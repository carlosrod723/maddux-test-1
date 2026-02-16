# query_maddux.py

"""
Claude API Integration

Natural language queries against the MADDUX database.
"""

import sqlite3
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "maddux_db.db")


def get_database_schema():
    """Return database schema for Claude's context."""
    return """
DATABASE SCHEMA:

TABLE: players
- player_id (INTEGER PRIMARY KEY) - MLBAM ID
- player_name (TEXT) - Full name

TABLE: player_seasons
- player_id (INTEGER) - FK to players
- year (INTEGER) - Season year
- pa (INTEGER) - Plate appearances
- max_ev (REAL) - Maximum exit velocity (mph)
- hard_hit_pct (REAL) - Hard hit percentage (95+ mph)
- barrel_pct (REAL) - Barrel percentage
- ops (REAL) - On-base plus slugging
- obp (REAL) - On-base percentage
- slg (REAL) - Slugging percentage
- wrc_plus (REAL) - Weighted runs created plus (100 = league avg)
- team (TEXT) - Team abbreviation

TABLE: player_deltas
- player_id (INTEGER) - FK to players
- year (INTEGER) - Season year (the "after" year)
- prev_year (INTEGER) - Previous season year
- pa (INTEGER) - Plate appearances in current year
- max_ev (REAL) - Current year max EV
- prev_max_ev (REAL) - Previous year max EV
- delta_max_ev (REAL) - Change in max EV
- hard_hit_pct (REAL) - Current year hard hit %
- prev_hard_hit_pct (REAL) - Previous year hard hit %
- delta_hard_hit_pct (REAL) - Change in hard hit %
- ops (REAL) - Current year OPS
- prev_ops (REAL) - Previous year OPS
- delta_ops (REAL) - Change in OPS
- maddux_score (REAL) - MADDUX Score = delta_max_ev + (2.1 × delta_hard_hit_pct)
- team (TEXT) - Current team

DATA RANGE: 2015-2025 seasons
TOTAL PLAYERS: 1,341
TOTAL PLAYER-SEASONS: 5,178
TOTAL DELTA RECORDS: 3,606 (consecutive year pairs)
"""


def execute_query(sql: str) -> list:
    """Execute SQL query and return results."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        results = [dict(row) for row in cursor.fetchall()]
        return results
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        conn.close()


def query_maddux(user_question: str) -> str:
    """Send natural language query to Claude, get SQL + analysis."""
    
    client = Anthropic()
    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    
    system_prompt = f"""You are a baseball analytics assistant with access to the MADDUX database.
Your job is to answer questions about MLB hitter performance and breakout predictions.

{get_database_schema()}

MADDUX MODEL EXPLANATION:
The MADDUX Hitter score predicts breakout potential based on year-over-year changes in physical metrics:
- Formula: MADDUX Score = Δ Max EV + (2.1 × Δ Hard Hit%)
- Theory: When underlying physical metrics improve, performance (OPS) typically follows
- High scores (>15) indicate significant physical improvement
- The 2.1 coefficient weights Hard Hit% changes more heavily than Max EV

INSTRUCTIONS:
1. Generate a valid SQLite query to answer the user's question
2. Return ONLY the SQL query wrapped in ```sql``` tags
3. After I execute it, I'll give you the results to analyze

Keep queries efficient. Use JOINs to get player names. Limit results to 20 unless asked for more."""

    # Step 1: Get SQL from Claude
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_question}
        ]
    )
    
    sql_response = response.content[0].text
    
    # Extract SQL from response
    import re
    sql_match = re.search(r'```sql\s*(.*?)\s*```', sql_response, re.DOTALL)
    
    if not sql_match:
        return f"Could not generate SQL query.\n\nClaude's response:\n{sql_response}"
    
    sql = sql_match.group(1).strip()
    
    # Step 2: Execute query
    results = execute_query(sql)
    
    # Step 3: Get analysis from Claude
    analysis_prompt = f"""The user asked: "{user_question}"

I ran this SQL query:
```sql
{sql}
```

Results ({len(results)} rows):
{json.dumps(results[:20], indent=2)}

Please provide a clear, insightful analysis of these results. Include:
1. Direct answer to the question
2. Key insights or patterns
3. Any caveats or limitations

Be concise but thorough. Use the MADDUX model context to explain what the numbers mean."""

    analysis_response = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[
            {"role": "user", "content": analysis_prompt}
        ]
    )
    
    analysis = analysis_response.content[0].text
    
    # Format output
    output = f"""
{'='*60}
QUERY: {user_question}
{'='*60}

SQL:
{sql}

RESULTS ({len(results)} rows):
{format_results(results[:15])}

{'='*60}
ANALYSIS:
{'='*60}
{analysis}
"""
    
    return output


def format_results(results: list) -> str:
    """Format results as a readable table."""
    if not results:
        return "No results found."
    
    if "error" in results[0]:
        return f"SQL Error: {results[0]['error']}"
    
    # Get column headers
    headers = list(results[0].keys())
    
    # Build table
    lines = []
    
    # Header row
    header_line = " | ".join(f"{h[:15]:<15}" for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Data rows
    for row in results:
        values = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                values.append(f"{v:<15.3f}")
            else:
                values.append(f"{str(v)[:15]:<15}")
        lines.append(" | ".join(values))
    
    return "\n".join(lines)


def interactive_mode():
    """Run interactive query session."""
    print("=" * 60)
    print("MADDUX Query Interface")
    print("=" * 60)
    print("Ask questions about MLB hitter breakouts and performance.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("\n🔍 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n⏳ Querying...")
        result = query_maddux(question)
        print(result)


def demo_queries():
    """Run demo queries to showcase the system."""
    print("=" * 60)
    print("MADDUX Query Demo")
    print("=" * 60)
    
    queries = [
        "Who are the top 10 players by MADDUX score in 2024?",
        "Show me players who had MADDUX scores above 20 and how their OPS changed",
        "What is Aaron Judge's MADDUX history?"
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"DEMO: {q}")
        result = query_maddux(q)
        print(result)
        print("\n" + "-"*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_queries()
    else:
        interactive_mode()
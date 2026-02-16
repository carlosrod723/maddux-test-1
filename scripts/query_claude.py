# query_claude.py

# ==============================================================================
# MADDUX Test 2: Claude API Integration
# ==============================================================================
# Gets top 10 players from database and sends to Claude for analysis.
# ==============================================================================

import sqlite3
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configuration
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "maddux_test.db")
OUTPUT_FILE = str(PROJECT_ROOT / "claude_response.txt")


def get_top_10():
    """Get top 10 players by MADDUX score from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
        SELECT 
            p.player_name,
            b.bb_pct,
            b.contact_pct,
            ROUND(b.bb_pct + (1.8 * b.contact_pct), 2) AS maddux_score
        FROM players p
        JOIN batting_2024 b ON p.player_id = b.player_id
        ORDER BY maddux_score DESC
        LIMIT 10
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    return results


def format_player_data(players):
    """Format player data for Claude prompt."""
    lines = ["Rank | Player | BB% | Contact% | Score"]
    lines.append("-" * 50)
    
    for i, (name, bb_pct, contact_pct, score) in enumerate(players, 1):
        lines.append(f"{i}. {name}: BB% {bb_pct:.1f}%, Contact% {contact_pct:.1f}%, Score {score:.2f}")
    
    return "\n".join(lines)


def query_claude(player_data):
    """Send prompt to Claude API and get response."""
    print("🤖 Querying Claude API...")
    
    client = Anthropic()
    
    prompt = f"""Here are the top 10 MLB hitters by plate discipline score (BB% + 1.8 × Contact%). Analyze their breakout potential for next season. Consider which players might improve their OBP based on these underlying metrics.

{player_data}"""

    model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    print(f"   Model: {model}")
    
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    response = message.content[0].text
    print("   ✓ Response received")
    
    return response


def save_response(response):
    """Save Claude's response to file."""
    with open(OUTPUT_FILE, 'w') as f:
        f.write(response)
    print(f"   ✓ Saved to {OUTPUT_FILE}")


def main():
    print("=" * 60)
    print("MADDUX Test 2: Claude API Integration")
    print("=" * 60 + "\n")
    
    # Get top 10 from database
    print("📊 Fetching top 10 players from database...")
    players = get_top_10()
    player_data = format_player_data(players)
    print(f"   ✓ Retrieved {len(players)} players\n")
    print(player_data)
    print()
    
    # Query Claude
    response = query_claude(player_data)
    
    # Save response
    save_response(response)
    
    # Print response
    print("\n" + "=" * 60)
    print("Claude's Analysis:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    print("\n✅ Complete! Response saved to claude_response.txt")


if __name__ == "__main__":
    main()
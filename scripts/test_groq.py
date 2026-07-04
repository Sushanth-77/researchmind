"""
Manual smoke test: confirms the Groq API key works and the model responds.

Run directly: python scripts/test_groq.py
Not a pytest test — this hits a real network API and costs nothing on
free tier, but shouldn't run automatically in CI.
"""

from groq import Groq

from researchmind.config import GROQ_API_KEY, GROQ_MODEL


def main() -> None:
    client = Groq(api_key=GROQ_API_KEY)

    print(f"Sending test prompt to Groq model: {GROQ_MODEL} ...")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly one sentence confirming you are working.",
            }
        ],
        max_tokens=50,
    )

    content = response.choices[0].message.content
    print("Groq response:")
    print(content)
    print("\n✅ Groq API connection confirmed.")


if __name__ == "__main__":
    main()
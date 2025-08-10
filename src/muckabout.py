from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI()

# Start the conversation
prompt = "Hello from VS Code — introduce yourself and tell me how fast you think you are."

for i in range(3):  # Run for 3 turns
    start = time.time()
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    end = time.time()

    output = response.output_text.strip()

    print(f"\n--- Turn {i+1} ---")
    print(f"Time taken: {round(end - start, 2)} seconds")
    print("Model's reply:", output)

    # Feed the response back as a playful follow-up
    prompt = f"Holy crap: '{output}'. Is an insane thing to say? Im horrified"
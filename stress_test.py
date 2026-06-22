from groq import Groq
import os

# Load API key
with open('groq.env') as f:
    for line in f:
        if line.strip() and '=' in line:
            key = line.split('=', 1)[1].strip()
            os.environ['GROQ_API_KEY'] = key

client = Groq(api_key=os.environ['GROQ_API_KEY'])

prompt = """You are helping test a label taxonomy for classifying r/nba posts. The labels are:

analysis: A structured argument backed by specific, verifiable evidence (statistics, historical comparison, tactical observation). The evidence is central to the post, not decorative.

hot_take: A bold, confident opinion stated with minimal or no supporting evidence. May cite a stat, but it is cherry-picked or decorative. The post asserts rather than argues.

reaction: An immediate emotional response to a specific event. Little to no argument or evidence — expressing a feeling in the moment.

Generate 10 posts from r/nba that sit at the BOUNDARY between two labels — posts that a human classifier might genuinely struggle to assign to exactly one label. Specifically:
- 4 posts at the boundary between analysis and hot_take
- 3 posts at the boundary between reaction and analysis
- 3 posts at the boundary between reaction and hot_take

For each post, briefly explain WHY it is ambiguous between those two labels.
Format each as:
---
Post: [the post text]
Boundary: [label A] / [label B]
Why ambiguous: [explanation]
---
"""

response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.7,
    max_tokens=1500
)

print(response.choices[0].message.content)

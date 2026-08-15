I would like you to create an app in which two AI’s debate each other, then a number of AI judges assess who wins by scoring in various categories. The whole thing should be able to be run with `docker compose up`. Here’s the details: The app should have a UI. 

1. The Debate setup phase. 
- The first step of the experience is the user entering the debate topic. The topic could be a general one like “Are driverless cars safe enough for public roads?”, or it could one related to recent news and events, for example: “Is the Australian government's current overhaul of the National Disability Insurance Scheme (NDIS) moving too far and too fast at the expense of vulnerable citizens?”. 
- Choose a model from huggingface - by default select a model from huggingface that does not have guardrails. If a model is already downloaded, it should say so in the UI.
- User clicks "Begin Debate". App downloads the model (if it hasn't already been downloaded)  

the rest of these steps are done automatically:

2. App goes off and researches the topic - scraping / collecting information - articles, wiki’s, ect.
3. Create 2 AI’s for debating the topic - one will debate FOR the topic, and the other AGAINST. (ie. could be MCP tools? - each get instructions saying they are subject matter experts ect… - they also have access to all the researched material (via a RAG pipeline, vector search or any means you think is most efficient)
4. The 2 debating AI’s then commence the debate in typical debate format. 
- Each get an opening statement
- Each get rounds to respond to each other sequentially
- Each get a closing statement
5. Create 3 AI judges (who are neutral) to judge all the debating AI’s responses to determine who won the debate. The judges should use a scoring system over several criteria, such as the standard 100-Point Ballot method.
6. Finally tally up the scores, and show who is the winner

- In the debate setup phase, add the ability for the user to adjust the personality / traits of the debating AI’s. For example, one could make them “proper and oxford-like”, or “sassy and sarcastic”. Perhaps make this is text box so we don’t limit the user’s creativity
- Load a single LLM model that all the AI’s use - factor in available RAM on the system, so it will download a version of the AI model that fits in memory.
- All the LLM processing can happen sequentially
- The user should able to watch the debate happen in real-time via the UI
- The user can export a PDF transcript of the debate at the end.

Finally, prepare the codebase so i can publish it on github.
# cobalt-mesa-geo — tiny Mesa-Geo + LLM-ready demo

## Quickstart
1. create venv
   python -m venv venv
   source venv/bin/activate

2. install
   pip install -r requirements.txt

3. generate demo data
   python data_gen.py

4. run demo (uses rule-based LLM fallback by default)
   python run.py

5. view map
   open logs/map.html in a browser

## If you have Ollama locally
- set `USE_OLLAMA=True` in `llm_wrapper.py` and `OLLAMA_MODEL` to your model
- make sure `ollama` CLI is in PATH

## Next steps
- add more agent types (traders, exporters, buyers)
- implement transport networks (networkx) and movement along routes
- swap in a real LLM and build richer prompt templates + caching
- add Mesa web server + mesa-geo visual modules to see live ticks

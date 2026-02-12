---
trigger: always_on
---

This workspace MUST run Python inside the micromamba prefix environment:
  /data/hongzefu/maniskillenv1114

Rules:
1) NEVER run `python`, `pip`, or `pytest` from the global PATH or the `base` env.
2) For any Python command, ALWAYS use one of the following forms:

   Preferred (no activation needed):
   - micromamba run -p /data/hongzefu/maniskillenv1114 -- python <args>
   - micromamba run -p /data/hongzefu/maniskillenv1114 -- pip <args>
   - micromamba run -p /data/hongzefu/maniskillenv1114 -- pytest <args>

   Acceptable:
   - /data/hongzefu/maniskillenv1114/bin/python <args>

3) Before running any long job, verify interpreter:
   - micromamba run -p /data/hongzefu/maniskillenv1114 -- python -c "import sys; print(sys.executable)"
   It must print: /data/hongzefu/maniskillenv1114/bin/python
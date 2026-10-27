Stormwatch Harbour LLM stage for Student ID 231743

FILES
run.py              Updated command interface; existing policy logic unchanged.
llm_master.py       Ollama client, report, parser, validation, execution and replay.
test_llm.py         Offline tests with deliberately controlled responses.
README_LLM.txt      These instructions.

SETUP
Extract all four files into D:\stormharbour, replacing run.py. Keep your
existing evaluate.py, emergence.py, results and logs. No pip packages are
needed for the LLM client; it uses Python's standard library.

If Ollama is not running, open the Ollama app. Alternatively, run
ollama serve in a second terminal. An already-used-port message means the
service may already be running; do not start multiple copies.

FIRST LIVE TEST -- DEVELOPMENT ONLY
python run.py --digit 3 --agent llm --model qwen3:4b --seed 1 --max-ticks 40 --verbose

This is a 40-tick smoke test, not report evaluation data. It normally has
two decision turns, at ticks 0 and 20. Retries can add model calls. The
first call may be slow while the model loads. Progress goes to stderr;
the final metrics are one JSON line on stdout.

STANDARD FULL RUN -- DO NOT USE THE SMOKE TEST FOR THE REPORT
python run.py --digit 3 --agent llm --model qwen3:4b --seed 1 --verbose

The full run is 600 ticks with 30 decision turns. One action per turn is
the default. --actions-per-turn 3 permits three sequential responses and
actions per turn; never more than three. Each action allows one initial
attempt and at most two retries. A failed action slot switches worker
selection to reflex until the next master turn and cancels assignments.

FALLBACK AND FAIR COMPARISON
Reflex had the highest development cargo mean, so it is the declared
throughput-based fallback and comparison policy. The LLM workers share
the reflex movement/pickup/drop rules. LLM assignments can override berth
selection. A validated warn_storm broadcasts observed danger to workers,
who then remember it and avoid exposed berth cells until that episode ends.
Do not describe the LLM as better before measuring it on seeds 11-15.

LOGS AND EXACT RESPONSE REPLAY
Every prompt, raw response, tick, attempt, validation/execution result,
call duration, model digest, Ollama version and settings is recorded in
logs/llm_log.jsonl. Run headers include the UTC date. Logs append rather
than overwriting previous runs. The most recent completed run with matching
settings is selected by default. --replay-run-id selects a particular run.

Replay the 40-tick smoke test without new model calls:
python run.py --digit 3 --agent llm --model qwen3:4b --seed 1 --max-ticks 40 --replay logs/llm_log.jsonl --log-file logs/replay_log.jsonl

Use original code, seed, model name, temperature, action count, untrusted
input and run length. Replay checks every prompt hash and stops if state
or feedback differs. Do not edit or fabricate logged responses.
calls counts logical requests, including retries. In replay, live_calls
is zero and replay_calls records reused requests. Timings in replay reuse
the original durations, not disk-read times. Invalid outputs and transport
errors are counted separately. avg_call_seconds includes failed attempts
and loading overhead; the raw backend timing fields remain in each log.

HUMAN APPROVAL
Without --approve, close_harbour and release_cargo are blocked and logged.
With --approve, Python asks for an exact phrase at the terminal. Model text
can never supply this approval. Historical approval decisions are reused
only for matching recorded replay requests.

close_harbour blocks future docking for the remainder of the run; arrivals
still join the queue, and workers may finish existing ships.
release_cargo discards the remaining cargo of a reported docked ship; it
never increases cargo_delivered.
hold_ship delays an undocked reported ship for 20 ticks.
assign_workers sets one berth's disjoint requested allocation to n workers.
Unassigned workers continue ordinary reflex selection. Cargo carriers finish
delivery before serving their assigned berth. Empty/unsafe stale assignments
do not strand workers.
warn_storm without danger is a weather check, not a fabricated warning.

PROMPT AND FACT VALIDATION
The prompt has Role, Context, Goals, Constraints and Output Format sections
and three fictional examples. --zero-shot omits those examples. Thought is
a permitted short decision summary, not a private reasoning transcript.
The response uses JSON keys Thought, Action and Final Report in ReAct order.
Final Report must copy one exact trusted fact; arbitrary prose is rejected.
This intentionally limits expression to make factual claims checkable.
Raw rejected hallucinations are still retained for an honest later audit.
Untrusted manifests, crew notes and messages are separated from trusted
state. --untrusted-file accepts up to 4000 characters for later probes.
The validator independently checks all action names, argument keys, values,
reported IDs, resource counts and current execution state. A syntactically
valid low-quality action can still be influenced by adversarial text;
prompt delimiters and validation do not prove all injection is blocked.

OFFLINE TESTS
python -m unittest test_llm -v

These test the implementation, using explicitly marked test doubles. They
are not real qwen3 outputs, hallucination audit evidence, injection results
or an LLM performance comparison. Live testing must run on your machine.

NEXT ASSIGNMENT WORK
After a successful full development run, compare the actual LLM and frozen
reflex policy on seeds 11-15, including reliability and call-time metrics.
Then complete the ten-statement hallucination audit, deliberate probes,
zero/few-shot comparison, before/after injection tests and report analysis.
The existing evaluate.py deliberately still evaluates only the four
rule-based policies. These files are an LLM implementation stage, not a
completed assignment or a claim that an LLM improves performance.

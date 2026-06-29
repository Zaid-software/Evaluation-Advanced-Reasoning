.PHONY: eval golden judge-check taxonomy demo clean

golden:
	python eval/generate_golden_set.py

eval: golden
	python -m eval.sanity_check_no_leakage
	python -m eval.run_eval
	python -m eval.compute_stats
	python -m eval.diff_baseline

judge-check:
	python -m eval.run_judge_sanity_check

taxonomy:
	python -m eval.failure_taxonomy

demo:
	python main.py --demo

clean:
	rm -f logs/trace_log.jsonl eval/raw_results.jsonl eval/stats_results.json eval/judge_sanity_check_results.json

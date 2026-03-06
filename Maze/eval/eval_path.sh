python eval/eval_path.py "8_test/result/0_result.json"
python eval/eval_path.py "16_test/result/0_result.json"
python eval/eval_path.py "32_test/result/0_result.json"

python eval/eval_path_overlap.py --level 8
python eval/eval_path_overlap.py --level 16
python eval/eval_path_overlap.py --level 32
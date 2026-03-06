LEVEL="level3"
NAME="0_result"

python eval/eval_path.py --table_dir "VSP/maps/level3/table" --json_path "VSP/maps/level3/img/result/${NAME}.json"
python eval/eval_path.py --table_dir "VSP/maps/level4/table" --json_path "VSP/maps/level4/img/result/${NAME}.json"
python eval/eval_path.py --table_dir "VSP/maps/level5/table" --json_path "VSP/maps/level5/img/result/${NAME}.json"
python eval/eval_path.py --table_dir "VSP/maps/level6/table" --json_path "VSP/maps/level6/img/result/${NAME}.json"
python eval/eval_path.py --table_dir "VSP/maps/level7/table" --json_path "VSP/maps/level7/img/result/${NAME}.json"
python eval/eval_path.py --table_dir "VSP/maps/level8/table" --json_path "VSP/maps/level8/img/result/${NAME}.json"

python eval/eval_path.py --table_dir "16_test/table" --json_path "16_test/result/0_result.json"
python eval/eval_path.py --table_dir "32_test/table" --json_path "32_test/result/0_result.json"


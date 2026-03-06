MODEL="Qwen-Image-Edit-2511"

python eval/diffthinker.py --level 12 --model "${MODEL}"
python eval/diffthinker.py --level 15 --model "${MODEL}"
python eval/diffthinker.py --level 18 --model "${MODEL}"

python eval/parse_image.py 12_test/result
python eval/parse_image.py 15_test/result
python eval/parse_image.py 18_test/result
# echo "Question-Only"
# echo "############# cn15k_baseline #############"
# python inference.py --dataset cn15k_baseline --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0
# echo "############# nl27k_baseline #############"
# python inference.py --dataset nl27k_baseline --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0
# echo "############# ppi5k_baseline #############"
# python inference.py --dataset ppi5k_baseline --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 0




# echo "Textual Graph random sample 50 tripels + Question"
# echo "############# cn15k_baseline #############"
# python inference.py --dataset cn15k_baseline --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 2096 --eval_batch_size 8
# echo "############# nl27k_baseline #############"
# python inference.py --dataset nl27k_baseline --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 2096 --eval_batch_size 8
# echo "############# ppi5k_baseline #############"
# python inference.py --dataset ppi5k_baseline --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 2096 --eval_batch_size 4



# echo "Textual Graph optimized by pcst + Question"
# echo "############# cn15k #############"
# python inference.py --dataset cn15k --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 2096 --eval_batch_size 4
# echo "############# nl27k #############"
# python inference.py --dataset nl27k --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 2096 --eval_batch_size 4
# echo "############# ppi5k #############"
# python inference.py --dataset ppi5k --model_name inference_llm --llm_model_name 7b_chat --max_txt_len 2096 --eval_batch_size 4
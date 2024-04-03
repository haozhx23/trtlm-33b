import random,os,time
import itertools

os.system('cp -r run_perf_trtlm_multigpu_gen.py TensorRT-LLM/examples/')


base_path='/home/ec2-user/SageMaker'
# base_path=/workspace
trt_engine_path=f'{base_path}/trt-lm-built-engines/chinese_llama33b_int8_engine'
test_tag='trt-deepseek-33b-int8'

def run_perf_test(glb_step, ilen=256, olen=128, bs=2):

    with open("en-novel.txt", "r") as file:
        content = file.read().replace('\n',' ')
        words = content.split(' ')

    with open('en-novel-test-batch.txt', "w") as file:
        for i in range(bs):
            random_start = random.randint(1, len(words)-ilen-10)
            test_str = " ".join(words[random_start: random_start+ilen+10])+'\n'
            file.write(test_str)
        
        
        
    cmdstr = f'''CUDA_VISIBLE_DEVICES=2,3 mpirun -n 2 \
        python TensorRT-LLM/examples/run_perf_trtlm_multigpu_gen.py \
                  --tokenizer_dir bloke-llama-tokenizer/ \
                  --engine_dir {trt_engine_path}/ \
                  --max_input_length {ilen} \
                  --max_output_len {olen} \
                  --input_file en-novel-test-batch.txt \
                  --profiling_tag {test_tag} \
                  --run_profiling'''

    print(perf_cmd)
    os.system(perf_cmd)



if __name__ == '__main__':

    ILEN = [538]
    OLEN = [1, 152]
    BS = [1,2,3,4]
    
    combinations = list(itertools.product(ILEN, OLEN, BS))
    tot_step = len(combinations)
    
    for glb_step, cb in enumerate(combinations):
        try:
            run_perf_test(glb_step, *cb)
            print(f'---DDD--- Global Step: {glb_step+1}/{tot_step} DONE.')

        except:
            print(f'---DDD--- Global Step: {glb_step+1}/{tot_step} FAIL.')
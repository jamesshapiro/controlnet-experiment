import subprocess

img_path = 'input/cat.png'

prompts = [
    'A lion staring at the camera. Best quality, ultra high res, artstation trending, highly detailed, (photorealistic:1.4)'
]

NUM_LOOPS = 4


prompt_values = []

prompts = sorted(prompts)

new_prompts = [prompt.replace('@me', 'zwx person') for prompt in prompts]


exec_prompts = []

for new_prompt in new_prompts:
    for i in range(NUM_LOOPS):
        exec_prompts.append(new_prompt)

for prompt in exec_prompts:
    #print(prompt)
    command = ['./build.sh', 'run', prompt, img_path, '--height', '768', '--width', '768', '--iters', '10', '--steps', '33']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    print(result.stderr.decode('utf-8'))
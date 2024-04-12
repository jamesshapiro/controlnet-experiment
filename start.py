import subprocess

prompts = [
    'hiyao miyazake man best quality, ultra high res, artstation trending, highly detailed,  greg rutkowski, thierry doizon, charlie bowater, alphonse mucha, dramatic lighting, (photorealistic:1.4)'
    # 'baseball card of @me, detailed textures, concept art, Greg Rutkowski',
    #'closeup portrait of @me as a superhero, symmetric face, artstation trending, highly detailed, hogwarts in the background, art by wlop, greg rutkowski, thierry doizon, charlie bowater, alphonse mucha, dramatic lighting, ultra realistic.',
    # 'closeup portrait of @me ukiyo-e, kyoto in the background, detailed textures, concept art, noir art, art by hinata matsumura, alphonse mucha, mike mignola, kazu kibuishi, and rev.matsuoka, digital painting, ultra-realistic.',
]


prompt_values = []

prompts = sorted(prompts)

new_prompts = [prompt.replace('@me', 'zwx person') for prompt in prompts]

NUM_LOOPS = 3
exec_prompts = []

for new_prompt in new_prompts:
    for i in range(NUM_LOOPS):
        exec_prompts.append(new_prompt)

for prompt in exec_prompts:
    #print(prompt)
    command = ['./build.sh', 'run', prompt, '--height', '768', '--width', '768', '--iters', '10', '--steps', '33']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))
    print(result.stderr.decode('utf-8'))
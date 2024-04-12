# Double-check DOCKERFILE for the entrypoint script, currently it is:

`ENTRYPOINT [ "docker-entrypoint.py" ]`

# You can modify the contents of docker-entrypoint.py by simply running:

`./build.sh`

# Running the program:

1. Input images should be placed in the "input/" folder
2. Set 'img_path', 'prompts' and 'NUM_LOOPS' in the start script
3. Finally run:

`python3 start`

This will run "./build.sh run ..."

Note that this completely crashes on larger images (> 15MB, many pixels), so smaller ones work much better

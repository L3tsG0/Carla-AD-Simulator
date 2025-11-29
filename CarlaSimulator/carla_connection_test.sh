export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python -c "import carla; client = carla.Client('localhost', 5555); print(client.get_server_version())"
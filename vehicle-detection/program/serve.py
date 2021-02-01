# This file implements the service shell. You don't necessarily need to modify it for various
# algorithms. It starts nginx and gunicorn with the correct configurations and then simply waits 
# until gunicorn exits.
#
# The flask server is specified to be the app object in wsgi.py
#
# We set the following parameters:
#
# Parameter                Environment Variable              Default Value
# ---------                --------------------              -------------
# number of workers        MODEL_SERVER_WORKERS              min(the number of CPU cores, DEFAULT_MAX_WORKERS)
# timeout                  MODEL_SERVER_TIMEOUT              60 seconds

import multiprocessing
import os
import signal
import subprocess
import sys

DEFAULT_MAX_WORKERS = 999
cpu_count = multiprocessing.cpu_count()

develop = os.environ.get('ENV', 'prod') == 'dev'
model_server_timeout = int(os.environ.get('MODEL_SERVER_TIMEOUT', 60))
max_model_server_workers = int(os.environ.get('MAX_MODEL_SERVER_WORKERS', DEFAULT_MAX_WORKERS))
default_model_server_workers = min(cpu_count, max_model_server_workers)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', default_model_server_workers))


def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


def start_server():
    print('Starting the inference server with {} workers.'.format(model_server_workers))

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/opt/program/nginx.conf'])
    reload_arg = ['--reload'] if develop else []
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers)] +
                                reload_arg +
                                ['wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    print('Inference server exiting')


# The main routine just invokes the start function.
if __name__ == '__main__':
    start_server()

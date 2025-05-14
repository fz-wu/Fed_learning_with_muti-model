import re
import subprocess
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os 
@csrf_exempt
def run_federated(request):
    if request.method == 'POST':
        print("post")
        # 直接从URL参数获取
        params = request.GET.dict()
        # 构造命令行参数
        cmd = ['python3', 'main.py']
        for k, v in params.items():
            cmd.append(f'--{k}')
            if v is not None and v != '':
                cmd.append(str(v))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Users/fafa/Documents/code/python/fedlearning')
            print("result", result.stdout)
            print("error", result.stderr)
            return JsonResponse({}, status=200)
        except Exception as e:
            return JsonResponse({}, status=500)
    else:
        return JsonResponse({'msg': '请使用POST请求提交参数。'}, status=200)


def get_models(request):
    if request.method == 'GET':
        models = os.listdir('/Users/fafa/Documents/code/python/fedlearning/models')
        return JsonResponse(models, safe=False)
    
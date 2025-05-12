import subprocess
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def run_federated(request):
    if request.method == 'POST':
        # 假设前端以 application/json 发送参数
        params = json.loads(request.body.decode())
        # 构造命令行参数
        cmd = ['python3', 'main.py']
        for k, v in params.items():
            cmd.append(f'--{k}')
            if v is not None and v != '':
                cmd.append(str(v))
        try:
            # 执行 main.py
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Users/fafa/Documents/code/python/fedlearning')
            return JsonResponse({
                'status': 'success' if result.returncode == 0 else 'fail',
                'stdout': result.stdout,
                'stderr': result.stderr,
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'msg': str(e)})
    else:
        # print("get")# 如果不是P
        return JsonResponse({'msg': '请使用POST请求提交参数。'}, status=200)
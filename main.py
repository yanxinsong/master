# 服务入口文件
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import numpy as np
import time
import pickle
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from typing import  List
app = FastAPI()
scheduler = AsyncIOScheduler()


model_path = "./checkpoints/model.pkl"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

def process_image(feature):
    feature = np.array(feature).reshape(1,36)
    lable = loaded_model.predict(feature)
    return int(lable[0])

@app.post("/predict")
async def predict(item_data: dict):
    # 获取json格式的参数
    feature = item_data.get("feature", [])
    id = item_data.get("id", None)
    if (not feature) or not isinstance(feature, List):
        return JSONResponse({"code": 1001, "message": "参数异常, 请检查参数"})
    if len(feature) != 36:
        return JSONResponse({"code": 1001, "message": "feature特征长度必须为36, 请检查参数"})
    try:
        print("***开始特征预测***")
        start = time.time()
        loop = asyncio.get_event_loop()
        label = await loop.run_in_executor(None, process_image, feature)
        end = time.time()
        print("result{}".format(label))
        print("***特征推测完成***")
    except Exception as e:
        print(e)
        return JSONResponse({"code": 1003, "message": "服务器异常"})
    # 将处理结果作为JSON响应返回
    return JSONResponse({
        "code": 200,
        "message": "特征预测成功",
        "result": {"lable":label, "id":id},
        "time": end - start,
    })


# 如果直接运行该脚本，则启动应用
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=81)

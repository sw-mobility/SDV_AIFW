import re

async def get_next_counter(
    mongo_client, 
    collection_name: str, 
    uid: str,  # uid 추가
    prefix: str,
    field: str, 
    width: int = 4
) -> str:
    cursor = mongo_client.db[collection_name].find({
        "uid": uid,  # uid 조건 추가
        field: {"$regex": f"^{prefix}\\d{{{width}}}$"}
    })
    list = [doc[field] for doc in await cursor.to_list(length=None)]
    max_num = 0
    for id in list:
        m = re.match(f"^{prefix}(\\d{{{width}}})$", id)
        if m:
            num = int(m.group(1))
            if num > max_num:
                max_num = num
    next_num = max_num + 1
    return f"{prefix}{str(next_num).zfill(width)}"
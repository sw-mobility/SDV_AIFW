from storage.managers.labeled.images.mongodb.file_manager import ImageFileMongoManager

def get_mongo_manager():
    return ImageFileMongoManager()

import redis

class MyRedis:
    def __init__(self):
        self.redisObject = redis.Redis(
        host='redis-13106.c89.us-east-1-3.ec2.redns.redis-cloud.com',
        port=13106,
        password='LZZSzxD7v4V3yvAhSwG1anlDgQHTJCwG')
        self.pubsub = self.redisObject.pubsub()
        self.pubsub.subscribe('messages')

    def get_pubsub(self):
        return self.pubsub
    
    def publish_message(self, message_type, sender, message):
        self.redisObject.publish('messages', f'{message_type}#{sender}#{message}')

if __name__ == '__main__':
    redisObject = MyRedis()



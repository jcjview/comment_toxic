import sys
import qiniu
access_key = 'JRDoymAJS0BLK1lvnmE7-m3fMNBhk3GtkdqKFHRt'
secret_key = 'xJU92iUwuGlsRa278Dbd3B6pjjbb29JYpk_ELnrG'
bucket_name='tnimage12121'#ucket_name 就是那个你创建的空间的名字
q = qiniu.Auth(access_key, secret_key)
print(sys.argv)
key = sys.argv[1]
print(key)
token = q.upload_token(bucket_name, key, 3600)
#要上传文件的本地路径
localfile = key
ret, info = qiniu.put_file(token, key, localfile)
if ret is not None:
    print('All is OK')
else:
    print(info) # error message in info


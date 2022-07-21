#!/usr/local/ancoonda3/envs/nas/bin/python
import urllib.parse
import urllib.request

url = 'http://www.pushplus.plus/send?'
token = '4be6998a8e3b429eb8617d55c47e6cb0'
title = 'Training_Status'
content = 'Training_Complete'
template = 'html'
request = url + 'token=' + token + '&title=' + title + '&content=' + content + '&template=' + template

print(request)

response = urllib.request.urlopen(request)

print(response)


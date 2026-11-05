"""Real HTTP collection for original callbacks and anonymous model metrics."""
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from validate_original_executor import observation_sink


def main():
    losses=[];posts=[]
    server=ThreadingHTTPServer(('127.0.0.1',0),observation_sink(losses,posts))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def send(path,body,headers):
        request=urllib.request.Request(base+path,data=body,headers=headers)
        with urllib.request.urlopen(request,timeout=3) as response:
            assert json.load(response)['code']==0
    try:
        send('/state/callback',b'{"state":"completed"}',{'Authorization':'synthetic-test-value'})
        assert posts==[dict(path='/state/callback',bytes=21,authenticated=True)]
        assert 'synthetic-test-value' not in json.dumps(posts)
        payload=b'loss{loss_name="loss",peer="1"} 0.25\n'
        send('/metrics/job/model',payload,{'Content-Type':'text/plain'})
        assert losses==[dict(value=.25,labels=dict(loss_name='loss',peer='1'))]
        try:send('/metrics/job/model',payload,{'Authorization':'synthetic-test-value'})
        except urllib.error.HTTPError as error:assert error.code==403
        else:raise AssertionError('authenticated validation metric was accepted')
        assert len(losses)==1 and len(posts)==2
        assert all(set(item)=={'path','bytes','authenticated'} for item in posts)
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)
    assert not thread.is_alive()
    print(json.dumps(dict(test='original_executor_observation_sink',original_callback_accepted=True,
        anonymous_metrics_parsed=True,headers_not_retained=True,cleanup_verified=True)))


if __name__=='__main__':main()

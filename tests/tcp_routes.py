"""Real loopback TCP aliases for node-local versus manager-visible BRPC routes."""
import asyncio
import threading


class TCPRoutes:
    def __enter__(self):
        self.loop=asyncio.new_event_loop();self.servers=[]
        self.thread=threading.Thread(target=self.loop.run_forever,daemon=True);self.thread.start()
        return self

    def route(self,target):
        host,port=target.rsplit(':',1)
        async def relay(reader,writer):
            while data:=await reader.read(65536):
                writer.write(data);await writer.drain()
            if writer.can_write_eof():writer.write_eof()
        async def connection(reader,writer):
            remote=None
            try:
                other,remote=await asyncio.open_connection(host,int(port))
                await asyncio.gather(relay(reader,remote),relay(other,writer))
            except (OSError,asyncio.CancelledError):pass
            finally:
                writer.close()
                if remote:remote.close()
        async def create():
            server=await asyncio.start_server(connection,'127.0.0.1',0)
            self.servers.append(server)
            return '127.0.0.1:'+str(server.sockets[0].getsockname()[1])
        return asyncio.run_coroutine_threadsafe(create(),self.loop).result(timeout=10)

    def __exit__(self,*_):
        async def close():
            for server in self.servers:server.close()
            for server in self.servers:await server.wait_closed()
            tasks=[t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        asyncio.run_coroutine_threadsafe(close(),self.loop).result(timeout=15)
        self.loop.call_soon_threadsafe(self.loop.stop);self.thread.join(timeout=5);self.loop.close()

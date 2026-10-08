import asyncio
import json
import time
from sglang_omni.client.client import Client, StreamedStopTrimmer
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.pipeline.coordinator import Coordinator
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

class InterleavedCoordinator(Coordinator):
    def __init__(self):
        super().__init__(completion_endpoint='inproc://complete', abort_endpoint='inproc://abort', entry_stage='preprocessing', terminal_stages=['decode','code2wav'])
        self.control_plane=RecordingCoordinatorControlPlane()
        self.register_stage('preprocessing','inproc://preprocessing')
    async def submit_request(self,request_id,request,*,stream_queue=None):
        await super().submit_request(request_id,request,stream_queue=stream_queue)
        await self.handle_stream(StreamMessage(request_id=request_id,from_stage='decode',modality='text',chunk={'text':'answer <ST','modality':'text'}))
        await self.handle_completion(CompleteMessage(request_id=request_id,from_stage='code2wav',success=True,result={'modality':'audio','sample_rate':24000}))
        await self.handle_stream(StreamMessage(request_id=request_id,from_stage='decode',modality='text',chunk={'text':'OP>','modality':'text'}))
        await self.handle_completion(CompleteMessage(request_id=request_id,from_stage='decode',success=True,result={'modality':'text','finish_reason':'stop'}))

async def run():
    request=GenerateRequest(prompt='hello',sampling=SamplingParams(stop=['<STOP>']),stream=True)
    chunks=[chunk async for chunk in Client(InterleavedCoordinator()).completion_stream(request,request_id='multimodal-stop')]
    return {'text':''.join(c.text or '' for c in chunks if c.modality=='text'),'expected':'answer ','chunks':[{'text':c.text,'modality':c.modality,'finish_reason':c.finish_reason} for c in chunks]}

report={'interleaved':asyncio.run(run()),'prefix_scan':[]}
for length in (1000,10000,100000,500000):
    trimmer=StreamedStopTrimmer(stop=['x '* (length//2)])
    start=time.perf_counter()
    returned=trimmer.push('hello')
    report['prefix_scan'].append({'stop_chars':length,'delta_chars':5,'seconds':time.perf_counter()-start,'returned':returned})
print(json.dumps(report,indent=2))

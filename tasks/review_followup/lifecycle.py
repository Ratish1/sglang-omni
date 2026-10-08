import json,multiprocessing,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path('/data/review-2604-2605-followup')
def await_file(path,timeout=45):
 start=time.monotonic()
 while not path.exists():
  if time.monotonic()-start>timeout:raise TimeoutError(str(path))
  time.sleep(.02)
def worker(case):
 directory=root/case
 (directory/'worker.pid').write_text(str(os.getpid()))
 import logging
 from types import SimpleNamespace
 from sglang_omni.pipeline import stage_workers
 import torch
 (directory/'before-entry').touch()
 await_file(directory/'gate')
 stage_workers.prepare_accelerator_environment=lambda *args:None
 stage_workers.apply_gpu_compat_env_defaults=lambda:None
 stage_workers.prepare_weight_share_process_compat=lambda:None
 def run_process(*args):
  allocation=torch.zeros(8*1024*1024,dtype=torch.uint8,device='cuda')
  torch.cuda.synchronize()
  (directory/'ready').write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'original_parent_alive':multiprocessing.parent_process().is_alive(),'gpu_bytes':allocation.numel()}))
  while True:time.sleep(.1)
 stage_workers.run_process=run_process
 stage_workers.stage_process_main(SimpleNamespace(stage_specs=[SimpleNamespace()],process_name='lifecycle-probe',log_level=logging.WARNING),None)
def alive(pid):
 try:return pathlib.Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
if __name__=='__main__':
 if len(sys.argv)>1:
  process=multiprocessing.get_context('spawn').Process(target=worker,args=(sys.argv[1],),daemon=True)
  process.start();process.join()
 else:
  rows=[]
  for case in ('parent-dies-after-entry','parent-dies-before-entry'):
   directory=root/case;directory.mkdir()
   parent=subprocess.Popen([sys.executable,__file__,case])
   child=None
   try:
    await_file(directory/'before-entry');child=int((directory/'worker.pid').read_text())
    if case=='parent-dies-after-entry':
     (directory/'gate').touch();await_file(directory/'ready')
    parent.kill();parent.wait()
    if case=='parent-dies-before-entry':(directory/'gate').touch()
    deadline=time.monotonic()+15
    while time.monotonic()<deadline and alive(child):time.sleep(.1)
    row={'case':case,'child_alive_after_parent_death':alive(child),'ready':json.loads((directory/'ready').read_text()) if (directory/'ready').exists() else None}
    rows.append(row);print(json.dumps(row),flush=True)
   finally:
    if parent.poll() is None:parent.kill();parent.wait()
    if child is not None and alive(child):os.kill(child,signal.SIGKILL)
  (root/'lifecycle-probe.json').write_text(json.dumps(rows,indent=2))

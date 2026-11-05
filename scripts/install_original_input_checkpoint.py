"""Install cursor support into a disposable original native source bundle.

Original bytes are retained outside the module. Reapplying verifies the previous
overlay, regenerates from those bytes and records exact input/output hashes.
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
FRAGMENTS=ROOT/'adapters/entropy/input_checkpoint_fragments'


def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('native source landmark is absent or ambiguous: '+before[:80])
    return text.replace(before,after,1)


def fragment(name):return (FRAGMENTS/(name+'.inc')).read_text()


def dependency(text,name,values):
    pattern=r'cc_library\(\s*name = "'+re.escape(name)+r'",[\s\S]*?\n\)'
    matches=list(re.finditer(pattern,text))
    if len(matches)!=1:raise ValueError('native BUILD target differs: '+name)
    match=matches[0];block=match.group()
    for value in values:
        if '"'+value+'"' not in block:
            block=replace(block,'deps = [','deps = [\n        "'+value+'",')
    return text[:match.start()]+block+text[match.end():]


def transform(original):
    result=dict(original)
    header='tensorswitch/file_feeder.h'
    text=replace(result[header],'#include "tensorflow/core/platform/file_system.h"',
        '#include "tensorflow/core/platform/file_system.h"\n#include "tensorswitch/eps_input_checkpoint.h"')
    declarations='''
eps::input_checkpoint::Cursor CaptureFileFeeder(tensorflow::RandomAccessFile* file);
tensorflow::Status RestoreFileFeeder(const std::string& stream,
    const eps::input_checkpoint::Cursor& cursor,
    std::unique_ptr<tensorflow::RandomAccessFile>* result);
'''
    result[header]=replace(text,'namespace tensorswitch {','namespace tensorswitch {\n'+declarations)
    filename='tensorswitch/file_feeder.cpp';text=result[filename]
    text=replace(text,'#include "tensorswitch/global.h"',
        '#include "tensorswitch/global.h"\n#include "tensorswitch/local_call.h"')
    text=replace(text,'  Status FetchNewFile(size_t current_file_size) const {',
        fragment('file_methods')+'  Status FetchNewFile(size_t current_file_size) const {')
    text=replace(text,'    const auto& filename = handle.response.filename();','''    const auto& filename = handle.response.filename();
    std::uint64_t assignment = eps::input_checkpoint::no_assignment;
    if (!filename.empty()) {
      try { assignment = eps::input_checkpoint::AssignmentId(handle.cntl.response_attachment().to_string()); }
      catch (const std::exception& error) { return errors::FailedPrecondition(error.what()); }
    }''')
    text=replace(text,'    if (file_switcher_) {','    assignment_ids_.push_back(assignment);\n    if (file_switcher_) {')
    text=replace(text,'  mutable std::vector<string> filenames_;',
        '  mutable std::vector<string> filenames_;\n  mutable std::vector<std::uint64_t> assignment_ids_;')
    text=replace(text,'  offset_ = 0;\n  return 0;','''  offset_ = 0;
  try { eps::input_checkpoint::RegisterSource(this, &filenames_, &offset_); }
  catch (const std::exception& error) { LOG(ERROR) << error.what(); return -1; }
  return 0;''')
    result[filename]=text+'\n'+fragment('native_functions')
    filename='tensorswitch/ps_service/ps_service_impl.cc';text=result[filename]
    text=replace(text,'#include "tensorswitch/global.h"',
        '#include "tensorswitch/global.h"\n#include "tensorswitch/eps_input_checkpoint.h"')
    text=replace(text,'  const auto& name = InputFiles::instance(handle)->Next();','''  std::pair<std::string, std::uint64_t> assignment;
  try {
    auto* source = InputFiles::instance(handle);
    assignment = eps::input_checkpoint::NextAssignment(source, [&]() -> const std::string& { return source->Next(); });
  } catch (const std::exception& error) { cntl->SetFailed(error.what()); return; }
  const auto& name = assignment.first;
  cntl->response_attachment().append(eps::input_checkpoint::AssignmentTag(assignment.second));''')
    result[filename]=text
    filename='tensorswitch/orc_dataset.cc';text=result[filename]
    text=replace(text,': DatasetIterator<Dataset>(params) {}','''
        : DatasetIterator<Dataset>(params) {
      checkpoint_reader_id_ = eps::input_checkpoint::RegisterReader(
          eps::input_checkpoint::ReaderScope(dataset()->filenames_));
    }
    ~Iterator() override { eps::input_checkpoint::ReaderDestroyed(checkpoint_reader_id_); }'''.lstrip())
    text=replace(text,'          *end_of_sequence = true;','''          auto completed = saved_cursor_;
          if (completed.stream.empty()) completed.stream = prefix();
          eps::input_checkpoint::ReaderCompleted(checkpoint_reader_id_, completed);
          *end_of_sequence = true;''')
    text=replace(text,'\n    next_row_index_ = 0;\n','\n    next_row_index_ = 0;\n    checkpoint_row_ = 0; checkpoint_filename_ = filename;\n')
    text=replace(text,'        next_row_index_ += 1;','        next_row_index_ += 1;\n        checkpoint_row_ += 1;')
    text=replace(text,'  Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors, size_t* read_bytes) {',
        fragment('orc_row_methods')+'  Status ReadRecord(IteratorContext* ctx, std::vector<Tensor>* out_tensors, size_t* read_bytes) {')
    text=replace(text,'  uint32_t next_row_index_ = 0;',
        '  uint32_t next_row_index_ = 0;\n  std::uint64_t checkpoint_row_ = 0;\n  std::string checkpoint_filename_;')
    pattern=r'    Status SaveInternal\([\s\S]*?(?=   private:)'
    found=list(re.finditer(pattern,text))
    if len(found)!=1:raise ValueError('original ORC iterator checkpoint methods differ')
    text=text[:found[0].start()]+fragment('orc_checkpoint')+text[found[0].end():]
    text=replace(text,'    void ResetStreamsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {',
        '''    void ResetStreamsLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (file_) {
        saved_cursor_ = tensorswitch::CaptureFileFeeder(file_.get());
        saved_cursor_.active = false; saved_cursor_.current = eps::input_checkpoint::no_assignment;
        saved_cursor_.filename.clear(); saved_cursor_.row = 0;
      }''')
    text=replace(text,'    mutex mu_;','    std::uint64_t checkpoint_reader_id_ = 0;\n    eps::input_checkpoint::Cursor saved_cursor_;\n    mutex mu_;')
    result[filename]=text
    text=dependency(result['BUILD'],'file_feeder',[':eps_input_checkpoint',':local_call'])
    text+='''
cc_library(
    name = "eps_input_checkpoint",
    hdrs = ["tensorswitch/eps_input_checkpoint.h"],
    strip_include_prefix = ".",
    visibility = ["//visibility:public"],
)
'''
    result['BUILD']=text
    text=result['tensorswitch/ps_service/BUILD']
    for name in ['ps_service_cpu','ps_service_gpu']:text=dependency(text,name,['//entropy:eps_input_checkpoint'])
    result['tensorswitch/ps_service/BUILD']=text
    result['tensorswitch/eps_input_checkpoint.h']=(ROOT/'adapters/entropy/input_checkpoint.h').read_text()
    return result


def sha(data):return hashlib.sha256(data).hexdigest()


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,pending=tempfile.mkstemp(prefix='.input-pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(pending,path)
    finally:
        pathlib.Path(pending).unlink(missing_ok=True)


def install(root,state):
    files=['BUILD','tensorswitch/file_feeder.h','tensorswitch/file_feeder.cpp',
        'tensorswitch/orc_dataset.cc','tensorswitch/ps_service/BUILD','tensorswitch/ps_service/ps_service_impl.cc']
    manifest=state/'manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    originals={}
    for name in files:
        backup=state/'original'/name
        if not backup.exists():atomic(backup,(root/name).read_bytes())
        data=backup.read_bytes()
        if previous and sha(data)!=previous['original_sha256'][name]:raise ValueError('original native overlay backup changed')
        originals[name]=data.decode()
    generated=transform(originals)
    original_hashes={f:sha(v.encode()) for f,v in originals.items()}
    generated_hashes={f:sha(v.encode()) for f,v in generated.items()}
    for name in generated:
        path=root/name
        if path.exists():
            digest=sha(path.read_bytes());allowed={original_hashes.get(name),generated_hashes[name]}
            if previous:
                allowed.add(previous['generated_sha256'].get(name))
                allowed.update(previous.get('accepted_sha256',{}).get(name,[]))
            if digest not in allowed:raise ValueError('native source changed outside the recorded overlay: '+name)
    # Publish intent first so an interrupted multi-file update is replayable.
    record=dict(kind='eps-original-input-overlay-v1',original_sha256=original_hashes,
        generated_sha256=generated_hashes,complete=False)
    record['accepted_sha256']={name:sorted({generated_hashes[name],
        *([original_hashes[name]] if name in original_hashes else []),
        *([previous['generated_sha256'][name]] if previous and name in previous['generated_sha256'] else []),
        *(previous.get('accepted_sha256',{}).get(name,[]) if previous else [])}) for name in generated}
    atomic(manifest,(json.dumps(record,indent=2)+'\n').encode())
    for name,text in generated.items():atomic(root/name,text.encode())
    record['complete']=True;atomic(manifest,(json.dumps(record,indent=2)+'\n').encode())
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=pathlib.Path,required=True,help='Disposable original Entropy module')
    parser.add_argument('--state',type=pathlib.Path,required=True,help='Private backup/manifest directory outside that module')
    args=parser.parse_args();root=args.root.resolve();state=args.state.resolve()
    if root==state or root in state.parents:parser.error('overlay state must be outside the source module')
    result=install(root,state)
    print(json.dumps(dict(complete=result['complete'],patched_files=len(result['generated_sha256']),manifest=str(state/'manifest.json'))))


if __name__=='__main__':main()

import os
from pathlib import Path

from ..log import RTAPDataLogger

class ETagStateMachine():
    def __init__(self, state_root, init_last_date=None):
        self.state_root = Path(state_root)
        self.etag_path = self.state_root / "etag.state"
        self.phish_path = self.state_root / "phish.state"
        self.benign_path = self.state_root / "benign.state"

        self.logger = RTAPDataLogger('etag-state-machine')
        if not self.state_root.exists():
            self.state_root.mkdir(parents = True, exist_ok = True)
            if init_last_date is not None:
                with open(self.etag_path, 'w') as fp:
                    fp.write(init_last_date)
        open(self.etag_path, 'a').close()        
        open(self.phish_path, 'a').close()
        open(self.benign_path, 'a').close()

    def get_etag(self):
        with open(self.etag_path, 'r') as fp:
            last_etag = fp.readline()
            return last_etag if last_etag else 0

    def update_etag(self, etag):
        with open(self.etag_path, 'w') as fp:
            fp.write(etag)

    def get_last_phish(self):
        with open(self.phish_path, 'r') as fp:
            last_phish_id = fp.read()
            return last_phish_id if last_phish_id else 0

    def update_last_phish(self, phish_id):
        with open(self.phish_path, 'w') as fp:
            fp.write(str(phish_id))

    def get_last_benign(self):
        with open(self.benign_path, 'r') as fp:
            last_id = fp.read()
            return int(last_id) if last_id else 0
        
    def generate_next_benigns(self, n):
        last_id = self.get_last_benign()
        next_ids = [i for i in range(last_id + n, last_id, -1)]
        return next_ids
    
    def update_last_benign(self, id):
        with open(self.benign_path, 'w') as fp:
            fp.write(str(id))
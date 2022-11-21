import numpy as np
ROOT = '<root>'
NULL = '<null>'
UNK = '<unk>'

def log_stats(sentences):
    num_tokens_per_s = list(map(lambda x: len(x.tokens), sentences))
    print('Token stats:')
    print(f'# of sentences: {len(sentences)}')
    print(f'mean: {np.mean(num_tokens_per_s)}, '
          f'median: {np.median(num_tokens_per_s)}, '
          f'std: {np.std(num_tokens_per_s)}')


class Token:
    def __init__(self, token_id, word, pos, head, dep):
        self.token_id = token_id
        self.word = word
        self.pos = pos
        self.head = head
        self.dep = dep
        self.lc, self.rc = [], []
        self.attached = False # Used with arc-eager system only
        
    def get_left_most_child(self, num=1):
        return self.lc[0 + num - 1] if len(self.lc) >= num else NULL_TOKEN
      
    def get_right_most_child(self, num=1):
        return self.rc[-1 - num + 1] if len(self.rc) >= num else NULL_TOKEN
            
    def __str__(self):
        return f"{self.token_id:5} | {self.word} | {self.pos} |" +\
            f" {self.head} | {self.dep}"

    def print_children(self):
        lc = [(t.word, t.token_id) for t in self.lc]
        rc = [(t.word, t.token_id) for t in self.rc]
        print(f'{self.token_id} {self.word} : left {lc} | right {rc}')
    
    def _enforce_children_order_debug(self):
        try:
            lc = [int(t.token_id) for t in self.lc]
            for i in range(len(lc) - 1):
                assert lc[i] < lc[i+1]
                assert int(self.token_id) > lc[i]
            if len(lc) > 1:
                assert int(self.token_id) > lc[-1]
        except:
            print('\nproblem with lc')
            breakpoint()
            return False
        try:
            rc = [int(t.token_id) for t in self.rc]
            for i in range(len(rc) - 1):
                assert rc[i] < rc[i+1]
                assert int(self.token_id) < rc[i]
            if len(rc) > 1:
                assert int(self.token_id) < rc[-1]
        except:
            print('\nproblem with rc')
            breakpoint()
            return False
        return True

    
        
    
# ROOT_TOKEN = Token(token_id="0", word=ROOT, pos=ROOT, head="-1", dep=ROOT)
NULL_TOKEN = Token(token_id="-1", word=NULL, pos=NULL, head="-1", dep=NULL)
UNK_TOKEN = Token(token_id="-1", word=UNK, pos=UNK, head="-1", dep=UNK)


class Sentence:

    def __init__(self, tokens=[], transition_system='std'):
        self.root = Token(token_id="0", word=ROOT, pos=ROOT, head="-1", dep=ROOT)
        self.tokens = tokens.copy()
        self.stack = [self.root]
        self.buffer = tokens.copy()
        self.arcs = []
        self.setup_trans_sys(transition_system)
    
    def setup_trans_sys(self, trans_system):
        if trans_system == 'std':
            # arc-standard
            self.update_state = self.update_state_std
            self._get_trans = self._get_trans_std
            self.supported_operations =\
                ['left_arc', 'right_arc', 'shift']
        else:
            # arc-eager
            self.update_state = self.update_state_eager
            self._get_trans = self._get_trans_eager
            self.supported_operations =\
                ['left_arc', 'right_arc', 'reduce', 'shift']
        
        
    def __len__(self):
        return len(self.tokens)
    
    def is_exausted(self):
        return len(self.stack) == 1 and len(self.buffer) == 0    
    
    def add_token(self, token):
        self.tokens.append(token)
        self.buffer.append(token)
        
    def peek_stack(self, top=1):
        items = list(reversed(self.stack[-top:]))
        if len(self.stack) < top:
            items += [NULL_TOKEN for _ in range(top - len(self.stack))]
        return items
    
    def peek_buffer(self, top=1):
        items =  self.buffer[:top]
        if len(self.buffer) < top:
            items += [NULL_TOKEN for _ in range(top - len(self.buffer))]
        return items

    def get_trans(self):  # this function is only used for the ground truth
        """ decide transition operation"""
        for operation in self.supported_operations:
            # Retrive transition name completely
            trans = self._get_trans(operation)
            if trans is not None:
                # print(trans, self)
                # breakpoint()
                return trans
        return None

    def _is_dep_in_buff(self, token_id):
        for t in self.buffer:
            if t.head == token_id:
                return True
        return False
    
    def _has_unassigned_child(self, token):
        for child_t in self.buffer + self.stack:
            if child_t in token.lc or child_t in token.rc:
                continue
            if child_t.head == token.token_id:
                return True
        return False

    def _get_trans_eager(self, potential_trans):
        """ get transition if it can legally be performed"""
        
        # LEFT, top of buffer is parent of top of stack
        def check_left_arc_sat():
            if self.buffer[0].token_id != self.stack[-1].head:
                return None
            return f"left_arc({self.stack[-1].dep})"

        # RIGHT, top of stack is parent of top in buffer       
        def check_right_arc_sat():
            if self.stack[-1].token_id != self.buffer[0].head:
                return None
            return f"right_arc({self.buffer[0].dep})"
        
        # top of the stack has an assigned parent and no unassigned children
        def check_reduce():
            if not self.stack[-1].attached or\
                self._has_unassigned_child(self.stack[-1]):
                return None
            return 'reduce'

        if len(self.stack) > 0 and len(self.buffer) > 0:
            if potential_trans == 'left_arc':
                return check_left_arc_sat()
            if potential_trans == 'right_arc':
                return check_right_arc_sat()
        if potential_trans == 'reduce' and len(self.stack) > 0:
            return check_reduce()
        if potential_trans == 'shift' and len(self.buffer) > 0:
            return 'shift'
        return None
    
    def update_state_eager(self, curr_trans, predicted_dep=None):
        """ 
        updates the sentence according to the given 
        transition assuming dependancy satisfiability
        but NOT legality
        """

        if 'shift' in curr_trans:
            if len(self.buffer) == 0:
                return False
            self.stack.append(self.buffer.pop(0))
            return True
        
        
        if 'reduce' in curr_trans:
            if len(self.stack) == 0:
                return False
            self.stack.pop(-1)
            return True
    
        if len(self.stack) < 1 or len(self.buffer) < 1:
            return False
        
        # top of buffer is parent of top of stack
        if 'left_arc' in curr_trans:
            parent = self.buffer[0]
            child = self.stack.pop(-1)
            # if not child.attached:
            parent.lc.insert(0, child)
            child.attached = True
            # DEBUG ONLY
            # breakpoint()
            # parent._enforce_children_order_debug()
            # if not parent._enforce_children_order_debug():
            #     breakpoint()
            # print('parent', parent.word)
            # breakpoint()
            if predicted_dep is not None:
                child.dep = predicted_dep
                child.head = parent.token_id
            self.arcs.append((parent, child, child.dep, 'l'))
            return True
            
        # top of stack is parent of top in buffer
        if 'right_arc' in curr_trans:
            parent = self.stack[-1]
            child = self.buffer[0]
            # if not child.attached:
            parent.rc.append(child)
            child.attached = True
            # DEBUG ONLY
            # if self.buffer[0].word == 'prevent':
            #     breakpoint()
            # parent._enforce_children_order_debug()
            #     breakpoint()
            # print('parent', parent.word)
            # breakpoint()
            if predicted_dep is not None:
                child.dep = predicted_dep
                child.head = parent.token_id
            self.arcs.append((parent, child, child.dep, 'r'))
        
            return self.update_state_eager('shift-r')

    def _get_trans_std(self, potential_trans):
        """ get transition if it can legally be performed"""
        # LEFT, top of stack is parent of second-top
        def check_left_arc_sat():
            if self.stack[-1].token_id != self.stack[-2].head:
                return None
            return f"left_arc({self.stack[-2].dep})"

        # RIGHT, second-top of stack is parent of top, 
        # and no depends of top in buffer (buff is empty)        
        def check_right_arc_sat():
            if self._is_dep_in_buff(self.stack[-1].token_id) or\
                self.stack[-2].token_id != self.stack[-1].head:
                return None
            return f"right_arc({self.stack[-1].dep})"

        if len(self.stack) >= 2:
            if potential_trans == 'left_arc':
                return check_left_arc_sat()
            if potential_trans == 'right_arc':
                return check_right_arc_sat()
        if potential_trans == 'shift' and len(self.buffer) >= 1:
            return 'shift'
        return None
    
    
    def update_state_std(self, curr_trans, predicted_dep=None):
        """ 
        updates the sentence according to the given 
        transition assuming dependancy satisfiability
        but NOT legality
        """
        
        if 'shift' in curr_trans:
            if len(self.buffer) == 0:
                return False
            self.stack.append(self.buffer.pop(0))
            return True
        
        if len(self.stack) < 2:
            return False
        
        if 'left_arc' in curr_trans:
            parent = self.stack[-1]
            child = self.stack.pop(-2)
            parent.lc.insert(0, child)
            if predicted_dep is not None:
                child.dep = predicted_dep
                child.head = parent.token_id
            self.arcs.append((parent, child, child.dep, 'l'))
            return True
            
        if 'right_arc' in curr_trans:
            parent = self.stack[-2]
            child = self.stack.pop(-1)
            parent.rc.append(child)
            if predicted_dep is not None:
                child.dep = predicted_dep
                child.head = parent.token_id
            self.arcs.append((parent, child, child.dep, 'r'))
            return True
        
    def __str__(self) -> str:
        return f"Stack {[t.word for t in self.stack]} | " +\
            f"Buffer {[t.word for t in self.buffer]}"
        
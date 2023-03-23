class RopeNode():
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.weight = len(value)
    
class Rope():
    def __init__(self, s):
        self.root = None
        for i in s:
            self.insert(i)
            
    def insert(self, value):
        node = RopeNode(value)
        if not self.root:
            self.root = node
            return
        
        current = self.root
        position = 0
        while True:
            if position <= current.weight:
                if not current.left:
                    self.split(current, node, position)
                    break
                else:
                    current = current.left
            else:
                position -= current.weight
                if not current.right:
                    self.split(current, node, position)
                    break
                else:
                    current = current.right
    
    def split(self, parent, node, position):
        left = RopeNode(parent.value[:position])
        right = RopeNode(parent.value[position:])
        
        left.left = parent.left
        left.right = node
        left.weight = len(left.value)
        
        right.left = node
        right.right = parent.right
        right.weight = len(right.value)
        
        parent.left = left
        parent.right = right
        parent.value = None
        parent.weight = left.weight + right.weight
        
    def traverse(self, node):
        if not node:
            return ""
        
        left = self.traverse(node.left)
        right = self.traverse(node.right)
        
        if not node.left and not node.right:
            return node.value
        else:
            return left + right
        

rope = Rope("This is a test.")
rope.insert(" ")
rope.insert("This is just a sample.")
result = rope.traverse(rope.root)
print(result)  # This is a test. This is just a sample.
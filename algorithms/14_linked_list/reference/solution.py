class ListNode:
    def __init__(self, value: int, next_node=None): self.value=value; self.next=next_node
def build_linked_list(values: list[int]):
    dummy=ListNode(0); tail=dummy
    for value in values: tail.next=ListNode(value); tail=tail.next
    return dummy.next
def reverse_linked_list(head):
    previous=None; current=head
    while current is not None:
        following=current.next; current.next=previous; previous=current; current=following
    return previous
def linked_list_values(head) -> list[int]:
    values=[]
    while head is not None: values.append(head.value); head=head.next
    return values
def main():
    n=int(input()); values=list(map(int,input().split())) if n else []
    if len(values)!=n: raise ValueError("节点数与 n 不一致")
    print(*linked_list_values(reverse_linked_list(build_linked_list(values))))
if __name__=="__main__": main()

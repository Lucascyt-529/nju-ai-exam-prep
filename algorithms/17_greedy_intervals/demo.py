import starter
def main():
    values=[(1,3),(2,6),(8,10),(10,12)]; expected=[(1,6),(8,12)]; print("输入:",values,"期望:",expected)
    try: print("实际:",starter.merge_intervals(values))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()

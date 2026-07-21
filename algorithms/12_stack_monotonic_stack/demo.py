import starter
def main():
    values=[73,74,75,71,69,72,76,73]; expected=[1,1,4,2,1,1,0,0]; print("输入:",values,"\n期望:",expected)
    try: print("实际:",starter.days_until_warmer(values))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()

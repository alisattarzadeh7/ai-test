from langchain_core.runnables import RunnableLambda


find_square = lambda x: x**2


runnable_sum = RunnableLambda(lambda x: sum(x))
runnable_square = RunnableLambda(lambda x: x**2)


chain = runnable_sum | runnable_square

print(chain.invoke([1,2,3]))
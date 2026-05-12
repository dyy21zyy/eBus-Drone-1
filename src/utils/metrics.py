def summarize(vals):
 return {"mean": sum(vals)/len(vals) if vals else 0.0, "n": len(vals)}

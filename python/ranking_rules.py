"""Predeclared ranking and calibration rules. Author: Sonja Sahebzad."""
import math
import numpy as np

VARIANTS=('baseline','no_boundary','half_boundary','token_mean','token_mean_no_boundary','char_sqrt','two_paths')
COLUMNS=('word','token_logp','boundary_logp','original_score','tokens','characters','alternate_score','alternate_distinct')


def scores(rows,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    result={}
    for word,l,b,total,n,c,alternate,distinct in rows:
        if variant=='baseline':v=total
        elif variant=='no_boundary':v=l
        elif variant=='half_boundary':v=l+.5*b
        elif variant=='token_mean':v=l/n+b
        elif variant=='token_mean_no_boundary':v=l/n
        elif variant=='char_sqrt':v=l/math.sqrt(c)+b
        else:v=float(np.logaddexp(total,alternate)) if distinct else total
        assert math.isfinite(v)
        result[word]=v
    assert len(result)==len(rows)
    return result


def prediction(rows,variant):
    values=scores(rows,variant)
    words=sorted(values,key=lambda w:(-values[w],w))
    maximum=values[words[0]]
    denominator=sum(math.exp(x-maximum) for x in values.values())
    return {'words':words[:3],'raw_shortlist_share':1/denominator,
        'top_score':maximum,'score_gap':maximum-values[words[1]]}


def metrics(details,variant):
    n=len(details)
    ranks=[d['predictions'][variant]['words'].index(d['actual'])+1 if d['actual'] in d['predictions'][variant]['words'] else 0 for d in details]
    return {'cases':n,'top1':sum(r==1 for r in ranks)/n,'top3':sum(r>0 for r in ranks)/n,
        'mrr_at3':sum(1/r if r else 0 for r in ranks)/n,
        'coverage':sum(d['covered'] for d in details)/n}


def sigmoid(x):
    return np.exp(-np.logaddexp(0.,-np.asarray(x,dtype=float)))


def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p)-np.log1p(-p)


def fit_sigmoid(raw,y):
    """Two-parameter logistic recalibration, slope penalty 0.5*slope^2.

    No ranking changes. Newton steps minimize sum log loss plus fixed penalty.
    """
    y=np.asarray(y,dtype=float)
    assert set(y)=={0.,1.}
    x=np.column_stack((np.ones(len(y)),logit(raw)))
    beta=np.array([float(logit(y.mean())),0.])
    penalty=np.diag([0.,1.])
    def objective(b):
        z=x@b
        return float(np.sum(np.logaddexp(0,z)-y*z)+.5*b[1]**2)
    for iteration in range(200):
        p=sigmoid(x@beta)
        gradient=x.T@(p-y)+penalty@beta
        if np.max(np.abs(gradient))<1e-7:break
        hessian=x.T@((p*(1-p))[:,None]*x)+penalty
        step=np.linalg.solve(hessian,gradient)
        scale=1.
        while objective(beta-scale*step)>objective(beta)-1e-4*scale*float(gradient@step):
            scale*=.5
            if scale<1e-10:raise RuntimeError('Calibration line search failed')
        beta-=scale*step
    else:raise RuntimeError('Calibration did not converge')
    return {'intercept':float(beta[0]),'slope':float(beta[1]),'iterations':iteration,
        'max_abs_gradient':float(np.max(np.abs(gradient))),'objective':objective(beta),
        'slope_l2_penalty':1.,'calibration_cases':len(y),'positives':int(y.sum())}


def calibrate(raw,model):
    return sigmoid(model['intercept']+model['slope']*logit(raw))


def wilson(successes,total):
    if not total:return [None,None]
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]


def reliability(y,p):
    y=np.asarray(y);p=np.asarray(p)
    rows=[]
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        n=int(mask.sum())
        rows.append({'bin':i+1,'lower':i/10,'upper':(i+1)/10,'cases':n,
            'mean_probability':float(p[mask].mean()) if n else None,
            'observed_accuracy':float(y[mask].mean()) if n else None})
    return rows


def confidence_metrics(y,p):
    y=np.asarray(y);p=np.clip(p,1e-12,1-1e-12)
    bins=reliability(y,p)
    return {'brier':float(np.mean((p-y)**2)),
        'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        'ece_10_equal_width':sum(r['cases']*abs(r['mean_probability']-r['observed_accuracy']) for r in bins if r['cases'])/len(y),
        'mean_probability':float(np.mean(p)),'observed_accuracy':float(np.mean(y)),
        'reliability':bins}

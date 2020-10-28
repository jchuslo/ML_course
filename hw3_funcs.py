'''This function scores 2 features of a decision tree based on
the information-loss metric/heuristic then prints both features and their heuristic scores, with features sorted by score'''

# Assume that "correct labeling" means that circles are
# classified in "true" and crosses are classified in "false"

circles = ['a','b','c','d']
crosses = ['w','x','y','z']

A_true = ['a','b']
A_false = ['c','d','w','x','y','z']
B_true = ['a','b','c','w']
B_false = ['d','x','y','z']

# tree
A = [A_true,A_false]
B = [B_true,B_false]
features = [A,B]

pos = len(circles)
neg = len(crosses)
total = pos + neg

H_examples = (pos/total)*math.log2(pos/total) + (neg/total)*math.log2(neg/total)
H_examples *= (-1)



entropy_list = []
for feat in features:
    H_funcs = []
    branch_totals = []
    for branch in feat:
        pos_p = len([x for x in branch if x in circles])
        neg_p = len([x for x in branch if x in crosses])

        if pos_p/len(branch) == 0:
            H_pos = pos_p/len(branch)
        else:
            H_pos = (pos_p/len(branch)) * math.log2(pos_p/len(branch))

        if neg_p/len(branch)==0:
            H_neg = neg_p/len(branch)
        else:
            H_neg = (neg_p/len(branch)) * math.log2(neg_p/len(branch))

        H = H_pos + H_neg
        H *= (-1)
        H_funcs.append(H)
        branch_total = len(branch)/total
        branch_totals.append(branch_total)
    
    remainder = sum([branch_totals[i]*H_funcs[i] for i in range(0,len(H_funcs))])
        
    gain = H_examples - remainder
    entropy_list.append(gain)
    
metrics = {"A":entropy_list[0],"B":entropy_list[1]}
m2 = sorted(metrics.items(), key=lambda x: x[1],reverse=True)

print("Sorted Features:")
print(f"Feature {m2[0][0]} : {metrics[m2[0][0]]:.3f}")
print(f"Feature {m2[1][0]} : {metrics[m2[1][0]]:.3f}")







'''This function computes the information-theoretic heuristic developed by Shannon for sample data on abalone shells'''

# TRAINING DATA

entropy_list={}
for feat in range(len(x_tr_s[0,:])):

    df = pd.DataFrame({'x':x_tr_s[:,feat], 'y':y_tr_s})
    data = np.asarray(df)

    a=[]
    b=[]
    for x in range(len(data)):
        if data[x,0]==0:
            a.append(data[x,1])
        elif data[x,0]==1:
            b.append(data[x,1])
    a = np.array(a)
    b = np.array(b)

    H_funcs = []
    branch_totals = []

    zero_tot = np.count_nonzero(y_tr_s==0)
    one_tot = np.count_nonzero(y_tr_s==1)
    two_tot = np.count_nonzero(y_tr_s==2)
    total = zero_tot + one_tot + two_tot

    H_examples = (zero_tot/total)*math.log2(zero_tot/total) + (one_tot/total)*math.log2(one_tot/total) + (two_tot/total)*math.log2(two_tot/total)
    H_examples *= (-1)

    for branch in [a,b]:
        zero_p = np.count_nonzero(branch==0)
        one_p = np.count_nonzero(branch==1)
        two_p = np.count_nonzero(branch==2)

        if zero_p/(zero_p+one_p+two_p)==0:
            H_0 = zero_p/(zero_p+one_p+two_p)
        else:
            H_0 = (zero_p/(zero_p+one_p+two_p)) * math.log2 ((zero_p/(zero_p+one_p+two_p)))

        if one_p/(zero_p+one_p+two_p)==0:
            H_1 = one_p/(zero_p+one_p+two_p)
        else:
            H_1 = (one_p/(zero_p+one_p+two_p)) * math.log2 (one_p/(zero_p+one_p+two_p))

        if two_p/(zero_p+one_p+two_p)==0:
            H_2 = two_p/(zero_p+one_p+two_p)
        else:
            H_2 = (two_p/(zero_p+one_p+two_p)) * math.log2 (two_p/(zero_p+one_p+two_p))

        H = H_0 + H_1 + H_2
        H *= (-1)

        H_funcs.append(H)
        branch_total = len(branch)/total
        branch_totals.append(branch_total)

    remainder = sum([branch_totals[i]*H_funcs[i] for i in range(0,len(H_funcs))])

    gain = H_examples - remainder
    entropy_list.update({feat:gain})
    

ent = sorted(entropy_list.items(), key=lambda x: x[1],reverse=True)

print("Sorted Features (starting at feature 0):")
for i in range(len(ent)):
    print(f"Feature {ent[i][0]} : {ent[i][1]:.3f}")
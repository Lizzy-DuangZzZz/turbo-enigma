from flask import Flask, request, jsonify
import numpy as np
from scipy import special as sp

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])

def calculate():
    #The equation for the orbital probability density. Please refer to Reference [1].
    def rho(r, q, n, l, m):
        N_r = np.sqrt(sp.factorial(n-l-1)*4.0/(np.power(n,4.0)*np.power(sp.factorial(n+l),1.0)))
        r = 2.0*r/n
        R = np.exp(-r/2.0)*np.power(r, l)*sp.genlaguerre(n-l-1, 2*l+1)(r)
        N_q = np.sqrt((2*l+1)*sp.factorial((l-np.abs(m)))/(4.0*np.pi*sp.factorial(l+np.abs(m))))
        Q = sp.lpmv(np.abs(m), l, np.cos(q))
        return np.power(N_r*R*np.abs(Q)*N_q, 2.0)
    
    data = request.get_json()
    a0 = 1.0
    n_a = data.get('n_a')
    N = data.get('N')
    n_p = data.get('n_p')
    n, l, m = data.get('n'), data.get('l'), data.get('m')
    #find the maximum probability
    r_max = np.sqrt(3.0*np.power(n_a*a0,2.0))
    R = np.linspace(0.0, r_max, n_a*25)
    Q = np.linspace(0.0, np.pi/2.0, 180)
    M_i = np.zeros(180)
    i=0
    for q in Q:
        M_i[i] = np.max(rho(R,q, n, l, m))
        i+=1
    M=1.05*np.max(M_i)
    print(M)
    p = 0
    #rnd_x = np.random.default_rng().uniform(-n_a*a0,0,N)
    rnd_x = np.random.default_rng().uniform(-n_a*a0,n_a*a0,N)
    rnd_y = np.random.default_rng().uniform(-n_a*a0,n_a*a0,N)
    rnd_z = np.random.default_rng().uniform(-n_a*a0,n_a*a0,N)
    r = np.sqrt(np.power(rnd_x,2.0)+np.power(rnd_y,2.0)+np.power(rnd_z,2.0))
    q = np.arccos(rnd_z/r)
    rho_xyz = rho(r,q, n, l, m)
    rnd_M = np.random.default_rng().uniform(0,M,N)
    #arrays to store the points
    x = np.zeros(n_p)
    y = np.zeros(n_p)
    z = np.zeros(n_p)
    for i in range(N):
        x_i, y_i, z_i = rnd_x[i], rnd_y[i], rnd_z[i]
        
        if rnd_M[i] <= rho_xyz[i]:
            x[p] = x_i
            y[p] = y_i
            z[p] = z_i
            p += 1
        if p+1 >= n_p:
            break
    print(p)
    data = np.zeros((p,3))
    data[:,0] = x[0:p]
    data[:,1] = y[0:p]
    data[:,2] = z[0:p]
    return jsonify({'result': data.tolist()})
    #np.savetxt("orbital_{:d},{:d},{:d}_full.csv".format(n,l,m), data, delimiter=",")
    del rnd_x, rnd_y, rnd_z, rnd_M, x, y, z
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

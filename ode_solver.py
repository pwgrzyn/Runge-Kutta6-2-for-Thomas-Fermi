import numpy as np 
from scipy.optimize import fsolve
class ode_solver:
    def __init__(self,system,init_cond,time,method='RK62',rtol=1e-4,beta=0.9):
        """ ODE Solver based on Runge-Kutta implicit adaptive step methods
        system: a right hand side function
        init_cond: list of initial conditions
        time: 2 element list of inital and final time
        method: indicates which Runge-Kutta method to use
        rtol: biggest possible error for which we accept the step size that produced such error between method and embedded method
        error: is taken as norm in Rn
        beta: coefficient that sclaes the step size
        """
        def _validate_system(system,init_cond):
            """ validating the input function and initial conditions"""
            result=system(time[0],init_cond)
            arr=np.atleast_1d(result)
            if arr.shape[0]!=len(init_cond):
                raise ValueError(f"Function output size {arr.shape[0]} does not match init_conditions size {len(init_cond)}")
            if arr.ndim != 1:
                raise ValueError("Function must return 1D array or list")
            return arr
        _validate_system(system,init_cond)
        self.init_cond=init_cond
        self.method=method
        self.sys_size=len(self.init_cond)
        self.rtol=rtol
        self.time_range=time[1]
        self.sol=[self.init_cond]
        self.time=[time[0]]
        self.beta=beta
        self.system=system
        np.seterr(invalid='raise')
        if(time[0]>=time[1]):
            raise ValueError("Final time has to be greater than starting")
        if(method not in ['RK62','RK43','Trap']):
            raise ValueError("There is no such method")
        if rtol<1e-14:
            raise ValueError("Minimal rtol is 1e-14}")
        if beta<=0:
            raise ValueError("Update coefficient can not be 0 or less") 
        if method=='RK62' :
            self.A=np.array([[5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],[5/36 + np.sqrt(15)/24, 
            2/9, 5/36 - np.sqrt(15)/24],[5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]],dtype=np.float64)
            self.b=np.array([5/18, 4/9,5/18],dtype=np.float64)
            self.b_tilda=np.array([-5/6,8/3,-5/6],dtype=np.float64)
            self.c=np.array([1/2 - np.sqrt(15)/10, 1/2,1/2 + np.sqrt(15)/10],dtype=np.float64)
            self.nstages=3
            self.order=6
            self.order_emb=2
        if method=='Trap':
            self.A=np.array([[0,0],[1/2,1/2]])
            self.c=np.array([0,1])
            self.b=np.array([1/2,1/2])
            self.b_tilda=np.array([1,0])
            self.nstages=2
            self.order=2
            self.order_emb=1
        if method=='RK43':
            self.A = np.array([
                [1/4,1/4 - np.sqrt(3)/6],
                [1/4 + np.sqrt(3)/6, 1/4          ]
            ], dtype=np.float64)
            self.b = np.array([1/2, 1/2], dtype=np.float64)
            self.b_tilda = np.array([1/2 + np.sqrt(3)/6, 1/2 - np.sqrt(3)/6], dtype=np.float64)
            self.c = np.array([1/2 - np.sqrt(3)/6, 1/2 + np.sqrt(3)/6], dtype=np.float64)
            self.nstages=2
            self.order=4
            self.order_emb=3
        self.solve()
    def _find_k(self,K,h):
        """ setting up equations for nonlinear solver"""
        eq=np.zeros(shape=(self.nstages,self.sys_size))
        K_res=K.reshape(self.nstages,self.sys_size)
        for stage in range(self.nstages):
            Y=self.sol[-1].copy()
            for j in range(self.nstages):
                Y+=h*self.A[stage,j]*K_res[j]
            eq[stage]=K_res[stage]-self.system(self.time[-1]+self.c[stage]*h,Y)
        return eq.flatten()
    def _evaluate_stages(self,K,h):
        """ solving equations for stages"""
        K=K.flatten()
        K_sol=fsolve(self._find_k,K,args=(h,))
        return K_sol.reshape(self.nstages, self.sys_size) 
    def _evaluate(self,K,h):
        """ evaluating next step of the system for both method and the embedded method"""
        K=np.array(K)
        K=K.reshape(self.nstages,self.sys_size)
        z = np.zeros(self.sys_size)
        z_tilda = np.zeros(self.sys_size)
        for i in range(self.sys_size):
            z[i]=np.sum(self.b*K[:,i])
            z_tilda[i]=np.sum(self.b_tilda*K[:,i])
        z_next=self.sol[-1]+h*z
        z_tilda=self.sol[-1]+h*z_tilda
        return [z_next,z_tilda]
    def _step_update(self,z_next,z_tilda,h,K):
        """ updating the step """
        step_error=np.linalg.norm(z_next-z_tilda)
        step_error=max(1e-32,step_error)
        n_iter=0
        while step_error>self.rtol and n_iter<100:
            h=self.beta*h*np.power((self.rtol/step_error),1/(self.order+1))
            h=max(h,1e-32)
            K=self._evaluate_stages(K,h)
            z_next,z_tilda=self._evaluate(K,h)
            step_error=np.linalg.norm(z_next-z_tilda)
            n_iter+= 1
        if self.time[-1]+h>self.time_range:
            h=self.time_range-self.time[-1]
            K=self._evaluate_stages(K,h)
            z_next,z_tilda=self._evaluate(K,h)
            t_curr=self.time_range
        else:
            t_curr=self.time[-1]+h
        h_new = self.beta * h * np.power((self.rtol / step_error), 1/(self.order))
        self.time.append(t_curr)
        self.sol.append(z_next)
        return h_new,K
    def solve(self):
        """ solve the equation/system of equations"""
        h=1/1000
        K=np.zeros((self.nstages,self.sys_size))  
        while self.time[-1]<self.time_range:
            K=self._evaluate_stages(K,h)
            z_next,z_tilda=self._evaluate(K,h)
            h,K=self._step_update(z_next,z_tilda,h,K)
        self.sol=np.array(self.sol)
        self.time=np.array(self.time)

        

from scipy.integrate import odeint
import numpy as np
import time
import sdeint

"""evolve module"""
class NoEquilibrium(Exception):
    pass

class evolve():
    def __init__( self, tipping_network, initial_state ):
        # Initialize solver
        self._net = tipping_network
        # Initialize state
        self._times = []
        self._states = []
        
        self._t = 0
        self._x = initial_state

        self.save_state( self._t, self._x ) 
        
    def save_state( self , t, x):
        """Save current state if save flag is set"""
        self._times.append( t )
        self._states.append( x )

    def get_timeseries( self ):
        times = np.array ( self._times )
        states = np.array ( self._states )
        return [times , states]
    
    def _integrate_sde( self, t_step,initial_state, sigma=None ):
        
        t_span = [ self._t , self._t + t_step ]
        x_init = self._x
         
        diffusion = lambda x,t: sigma 
        sol=sdeint.itoint(self._net.f,diffusion,x_init,t_span)
        self._t = t_span[1]
        
        self._x = sol[1]        
        self.save_state(self._t, self._x)    
        
    def _integrate_ode( self, t_step):
        
        t_span = [ self._t , self._t + t_step ]
        x_init = self._x
        
        sol = odeint( self._net.f , x_init, t_span, Dfun=self._net.jac )
            
        self._t = t_span[1]

        self._x = sol[1]
        
        self.save_state(self._t, self._x)
        
    def integrate( self, t_step, t_end,initial_state, sigma=None ):
        """Manually integrate to t_end"""
        
        if sigma is None:
            while self._times[-1] < t_end:
                self._integrate_ode( t_step )
        else:
            while self._times[-1] < t_end:
                    self._integrate_sde( t_step,initial_state,sigma )
    
    def equilibrate( self, tol , t_step, t_break=None,sigma=None ):
        """Iterate system until it is in equilibrium. 
        After every iteration it is checked if the system is in a stable
        equilibrium"""
        t0 = time.process_time()
        
        if sigma is None:
            while not self.is_equilibrium( tol ): 
                self._integrate_ode( t_step,sigma )
                if t_break and (time.process_time() - t0) >= t_break:
                    raise NoEquilibrium(
                            "No equilibrium found " \
                            "in " + str(t_break) + " seconds." \
                            " Increase tolerance or breaktime."
                            )
        else:
            while not self.is_equilibrium( tol ): 
                self._integrate_sde( t_step,sigma )
                if t_break and (time.process_time() - t0) >= t_break:
                    raise NoEquilibrium(
                            "No equilibrium found " \
                            "in " + str(t_break) + " seconds." \
                            " Increase tolerance or breaktime."
                            )
   
    def is_equilibrium( self, tol ):
        """Check if the system is in an equilibrium state, e.g. if the 
        absolute value of all elements of f (f is x_dot) is less than tolerance. 
        If True the state can be considered as close to a fixed point"""
        n = self._net.number_of_nodes()
        f = self._net.f( self._x, self._t)
        fix = np.less( np.abs(f) , tol * np.ones( n ))
        
        if fix.all():
            return True
        else:
            return False

    def is_stable( self ):
        """Check stability of current system state by calculating the 
        eigenvalues of the jacobian (all eigenvalues < 0 => stable)."""
        n = self._net.number_of_nodes()
        jacobian = self._net.jac( self._x, self._t)
        val, vec = np.linalg.eig( jacobian )
        stable = np.less( val, np.zeros( n ) )

        if stable.all():
            return True
        else:
            return False

    def get_autocorrelation( self , start_point, 
            detrend_window=1000, step_size=10):
        
        #print("Total length of time steps: ", len(self._times))
        #print("First timestep to start calculating autocorrelation: ", start_point)
        n = self._net.number_of_nodes()
        t = np.array(self._times[start_point:])
        x = np.array(self._states[start_point:])
        N = len(t)  # length of dataset after start point
        M = (N-detrend_window)//step_size # length of autocorrelation values
        #print("Number of time steps after starting point: ", N)
        #print("Size of detrending window: ", detrend_window)
        #print("Number of autocorrelation values to be calculated", M)
        #print("Length of iteration: ", len(range(0, N-detrend_window, step_size)))

        autocorr = np.zeros(( M,n ))
        count = 0
        
        for i in range(0, M*step_size, step_size): 
            
            for node in range(0,n):
                
                # Detrend the state values within the detrend window
                # for each node (should these be different bc of timescales?)
                trend = np.polyval ( np.polyfit(t[i:i+detrend_window], 
                                                x[i:i+detrend_window,node], 1), 
                                                t[i:i+detrend_window])
                x_detrend = x[i:i+detrend_window,node] - trend
                 
                # Calculate correlation coefficient with lag 1
                coeff_lag1 = np.corrcoef(x_detrend[:-1],x_detrend[1:])[0,1]
                
            
                autocorr[count,node] = coeff_lag1
            #print(count)
            #print(i+detrend_window)
            count += 1
        
        end_point = start_point + M*step_size
        #print("Last time starting point for autocorrelation to be calculated: ", end_point)
        #print ("Time array length: ", len(self._times[start_point:end_point:step_size]))
        return autocorr, end_point







            
        



                

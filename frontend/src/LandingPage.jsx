import React, { useState, useEffect } from 'react';
import { motion, useAnimation } from 'framer-motion';

// --- SVGs & Icons ---
const LogoIcon = ({ className = "w-10 h-10" }) => (
  <div className={`${className} relative group overflow-hidden bg-slate-900 border border-slate-700 rounded-xl flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.2)]`}>
    <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" className="w-6 h-6 relative z-10 drop-shadow-md">
      <path d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zm-2.625 6c-.54 0-.975.435-.975.975v.015c0 .54.435.975.975.975h5.25c.54 0 .975-.435.975-.975V9.225c0-.54-.435-.975-.975-.975h-5.25zm0 4.5c-.54 0-.975.435-.975.975v.015c0 .54.435.975.975.975h5.25c.54 0 .975-.435.975-.975v-.015c0-.54-.435-.975-.975-.975h-5.25z" />
    </svg>
  </div>
);

// --- Advanced Animation Variants (Spring Physics) ---
const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 100, damping: 20 } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.1 }
  }
};

const floatPhysics = {
  animate: {
    y: [0, -8, 0],
    rotate: [0, 1, -1, 0],
    transition: { duration: 6, repeat: Infinity, ease: "easeInOut" }
  }
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.9, filter: "blur(10px)" },
  visible: { opacity: 1, scale: 1, filter: "blur(0px)", transition: { duration: 0.7, ease: "easeOut" } }
};

// --- Main Component ---
const LandingPage = ({ onAccessDashboard }) => {
  const [contactStatus, setContactStatus] = useState('');
  const [scrolled, setScrolled] = useState(false);

  // Dynamic Navbar background on scroll
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleContactSubmit = async (e) => {
    e.preventDefault();
    setContactStatus('Deploying...');
    setTimeout(() => setContactStatus('Message Intercepted.'), 1500); 
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-200 font-sans overflow-x-hidden selection:bg-indigo-500/50">
      
      {/* --- ELITE BACKGROUND FX --- */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        {/* Animated Grid */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-50"></div>
        {/* Ambient Glows */}
        <div className="absolute top-[-20%] left-[-10%] w-[800px] h-[800px] bg-indigo-900/20 rounded-full blur-[150px]"></div>
        <div className="absolute top-[40%] right-[-10%] w-[600px] h-[600px] bg-blue-900/10 rounded-full blur-[150px]"></div>
      </div>

      {/* --- DYNAMIC HEADER --- */}
      <header className={`fixed top-0 left-0 right-0 z-50 border-b transition-all duration-300 ${scrolled ? 'bg-[#020617]/80 backdrop-blur-xl border-white/10 shadow-lg' : 'bg-transparent border-transparent'}`}>
        <nav className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="flex items-center space-x-4 cursor-pointer">
            <LogoIcon />
            <span className="text-xl font-black tracking-tight text-white drop-shadow-md">Nawthviper</span>
          </motion.div>
          
          <div className="hidden md:flex items-center space-x-8">
            {['Architecture', 'Engine', 'Terminal', 'Deploy'].map((item, idx) => (
              <motion.a 
                initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 * idx }}
                key={item} href={`#${item.toLowerCase()}`} 
                className="text-sm font-bold text-slate-400 hover:text-white transition-colors relative group"
              >
                {item}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-indigo-500 transition-all duration-300 group-hover:w-full"></span>
              </motion.a>
            ))}
          </div>

          <motion.button 
            initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
            onClick={onAccessDashboard} 
            className="hidden sm:block relative overflow-hidden group bg-indigo-600 text-white text-sm font-bold py-2.5 px-6 rounded-lg transition-all shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_30px_rgba(79,70,229,0.5)] active:scale-95"
          >
            <span className="relative z-10">Access Terminal</span>
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </motion.button>
        </nav>
      </header>

      <main className="relative z-10">
        
        {/* --- HERO SECTION --- */}
        <section id="architecture" className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 px-6">
          <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
            
            {/* Left: Typography & Action */}
            <motion.div initial="hidden" animate="visible" variants={staggerContainer} className="text-left">
              <motion.div variants={fadeInUp} className="inline-flex items-center space-x-3 bg-slate-900/50 backdrop-blur-md border border-slate-700/50 rounded-full px-4 py-2 mb-8 shadow-inner">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500"></span>
                </span>
                <span className="text-xs font-black tracking-widest uppercase text-slate-300">V2.0 Core Deployed</span>
              </motion.div>
              
              <motion.h1 variants={fadeInUp} className="text-5xl lg:text-7xl font-black tracking-tight leading-[1.05] mb-6 text-white">
                Algorithmic <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-blue-400 to-indigo-400 bg-[length:200%_auto] animate-gradient-x">
                  Supremacy.
                </span>
              </motion.h1>
              
              <motion.p variants={fadeInUp} className="text-lg text-slate-400 mb-10 max-w-lg leading-relaxed font-medium">
                Human emotion is a liability. Nawthviper connects directly to your MT5 terminal to execute institutional-grade price action models with mathematical violence.
              </motion.p>
              
              <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-5">
                <button onClick={onAccessDashboard} className="relative group bg-white text-[#020617] font-black py-4 px-8 rounded-xl transition-all hover:bg-slate-200 active:scale-95 overflow-hidden flex items-center justify-center">
                  <span className="relative z-10 flex items-center">
                    Initialize Engine
                    <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M17 8l4 4m0 0l-4 4m4-4H3" /></svg>
                  </span>
                </button>
                <a href="#engine" className="px-8 py-4 rounded-xl border border-slate-700 hover:bg-slate-800 text-slate-300 hover:text-white font-bold transition-all text-center flex items-center justify-center">
                  View Architecture
                </a>
              </motion.div>
            </motion.div>

            {/* Right: Abstract Interactive Terminal */}
            <motion.div initial="hidden" animate="visible" variants={scaleIn} className="relative hidden lg:block perspective-1000">
              <motion.div variants={floatPhysics} animate="animate" className="relative z-10 w-full max-w-[500px] ml-auto">
                
                {/* Main Glass Panel */}
                <div className="bg-[#0f172a]/80 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden">
                  {/* Top Bar */}
                  <div className="h-10 bg-[#1e293b]/50 border-b border-white/5 flex items-center px-4 space-x-2">
                    <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
                    <div className="ml-4 text-[10px] font-mono text-slate-500 tracking-widest">NAWTHVIPER_EXECUTION_NODE</div>
                  </div>
                  
                  {/* Terminal Content */}
                  <div className="p-6 space-y-5 font-mono">
                    <div className="flex justify-between items-center pb-4 border-b border-slate-700/50">
                      <div className="flex items-center space-x-3">
                        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]"></div>
                        <span className="text-xs text-slate-300">Connection</span>
                      </div>
                      <span className="text-xs font-bold text-emerald-400">5.58 ms (Direct)</span>
                    </div>

                    <div className="space-y-3">
                      <div className="text-xs text-slate-500">[SYSTEM] Awaiting tick data...</div>
                      <div className="text-xs text-indigo-400">[H4] Bias ALIGNED: UPTREND</div>
                      <div className="text-xs text-slate-300">[M1] Evaluating CRT Sweep on XAUUSD...</div>
                      
                      {/* Animated Progress Bar */}
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden mt-2">
                        <motion.div 
                          className="h-full bg-indigo-500"
                          initial={{ width: "0%" }}
                          animate={{ width: "100%" }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        />
                      </div>
                    </div>

                    <div className="pt-4 flex gap-3">
                       <div className="flex-1 bg-[#020617] border border-slate-800 p-3 rounded-lg">
                          <div className="text-[10px] text-slate-500 mb-1">CAPITAL GUARD</div>
                          <div className="text-sm font-bold text-emerald-400">ACTIVE</div>
                       </div>
                       <div className="flex-1 bg-[#020617] border border-slate-800 p-3 rounded-lg">
                          <div className="text-[10px] text-slate-500 mb-1">LOT SIZE</div>
                          <div className="text-sm font-bold text-white">DYNAMIC</div>
                       </div>
                    </div>
                  </div>
                </div>

                {/* Floating Badge */}
                <motion.div 
                   initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 1 }}
                   className="absolute -right-12 top-20 bg-slate-800/90 backdrop-blur-md border border-slate-600 p-4 rounded-2xl shadow-xl flex items-center space-x-4"
                >
                   <div className="bg-indigo-500/20 p-2 rounded-lg text-indigo-400">
                     <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                   </div>
                   <div>
                     <div className="text-xs text-slate-400 font-bold uppercase">Latency</div>
                     <div className="text-lg font-black text-white">0.4ms</div>
                   </div>
                </motion.div>

              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* --- FEATURES (ADVANCED BENTO GRID) --- */}
        <section id="engine" className="py-24 relative border-y border-white/5 bg-[#070b19]">
          <div className="max-w-7xl mx-auto px-6">
            <motion.div initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={staggerContainer} className="mb-20">
              <motion.h2 variants={fadeInUp} className="text-4xl md:text-5xl font-black mb-6 text-white tracking-tight">
                Designed for <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-blue-400">War.</span>
              </motion.h2>
              <div className="h-1.5 w-24 bg-gradient-to-r from-indigo-500 to-transparent rounded-full"></div>
            </motion.div>

            {/* The Grid */}
            <motion.div 
               initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={staggerContainer}
               className="grid grid-cols-1 md:grid-cols-3 gap-6 auto-rows-[minmax(220px,auto)]"
            >
              
              {/* Feature 1: Large Panel */}
              <motion.div variants={fadeInUp} className="md:col-span-2 group relative bg-slate-900 border border-slate-800 hover:border-indigo-500/50 rounded-3xl p-8 overflow-hidden transition-all duration-500 shadow-lg">
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-600/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <h3 className="text-2xl font-bold mb-4 text-white relative z-10">Institutional Logic</h3>
                <p className="text-slate-400 max-w-lg relative z-10 font-medium leading-relaxed">
                  The engine ignores retail indicators. It maps H4 structural zones using Numba optimization and executes strictly on M1 liquidity sweeps and momentum continuation.
                </p>
                <div className="mt-10 flex flex-wrap gap-3 relative z-10">
                   <span className="px-4 py-2 bg-[#020617] rounded-xl text-xs font-mono font-bold text-slate-300 border border-slate-700 shadow-inner">Numba Accelerated</span>
                   <span className="px-4 py-2 bg-[#020617] rounded-xl text-xs font-mono font-bold text-slate-300 border border-slate-700 shadow-inner">Pure Price Action</span>
                   <span className="px-4 py-2 bg-[#020617] rounded-xl text-xs font-mono font-bold text-indigo-400 border border-indigo-900/50 shadow-inner">Multi-Threaded</span>
                </div>
              </motion.div>

              {/* Feature 2: Tall Vertical Panel */}
              <motion.div variants={fadeInUp} className="md:row-span-2 group relative bg-slate-900 border border-slate-800 hover:border-red-500/40 rounded-3xl p-8 flex flex-col justify-between overflow-hidden transition-all duration-500 shadow-lg">
                <div className="absolute top-0 right-0 w-64 h-64 bg-red-600/5 rounded-full blur-3xl group-hover:bg-red-600/10 transition-all duration-500"></div>
                <div className="relative z-10">
                   <div className="w-14 h-14 bg-[#020617] border border-slate-700 rounded-2xl flex items-center justify-center mb-8 text-red-400 shadow-inner group-hover:scale-110 transition-transform duration-500">
                     <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" /></svg>
                   </div>
                   <h3 className="text-2xl font-bold mb-4 text-white">Capital Guard</h3>
                   <p className="text-slate-400 text-sm font-medium leading-relaxed">
                     Hard-coded risk limits protect your account from systemic drawdowns. Set your daily loss limit and the server triggers an immutable kill-switch. No tilt, no blown accounts.
                   </p>
                </div>
                <div className="mt-8 pt-8 border-t border-slate-800 relative z-10">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                      <p className="text-xs font-mono font-bold text-red-400">CIRCUIT_BREAKER = ACTIVE</p>
                    </div>
                </div>
              </motion.div>

              {/* Feature 3: Standard Panel */}
              <motion.div variants={fadeInUp} className="group bg-slate-900 border border-slate-800 hover:border-emerald-500/40 rounded-3xl p-8 transition-all duration-500 shadow-lg">
                <h3 className="text-xl font-bold mb-3 text-white">Dynamic Sizing</h3>
                <p className="text-sm text-slate-400 font-medium leading-relaxed">
                  Automatically scales position volumes based on real-time equity and precise stop-loss pip distance. Risk is locked to a mathematical constant.
                </p>
              </motion.div>

              {/* Feature 4: Standard Panel */}
              <motion.div variants={fadeInUp} className="group bg-slate-900 border border-slate-800 hover:border-blue-500/40 rounded-3xl p-8 transition-all duration-500 shadow-lg">
                 <h3 className="text-xl font-bold mb-3 text-white">Session Filters</h3>
                 <p className="text-sm text-slate-400 font-medium leading-relaxed">
                   The engine sleeps during the Asian session. It locks out execution during low-liquidity hours to prevent spread-widening and retail fakeouts.
                 </p>
              </motion.div>

            </motion.div>
          </div>
        </section>

        {/* --- DEPLOYMENT / CONTACT FORM --- */}
        <section id="deploy" className="py-32 relative">
           <div className="absolute inset-0 bg-gradient-to-b from-transparent to-[#020617] pointer-events-none"></div>
           <div className="max-w-xl mx-auto px-6 relative z-10">
              <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer} className="text-center mb-12">
                 <motion.div variants={fadeInUp} className="w-16 h-16 bg-indigo-600/20 border border-indigo-500/50 rounded-2xl mx-auto flex items-center justify-center mb-6">
                   <svg className="w-8 h-8 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
                 </motion.div>
                 <motion.h2 variants={fadeInUp} className="text-4xl font-black text-white tracking-tight">System Deployment</motion.h2>
                 <motion.p variants={fadeInUp} className="text-slate-400 mt-4 font-medium">Require integration assistance or custom architecture? Transmit a secure message.</motion.p>
              </motion.div>
              
              <motion.form 
                 initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.3 }}
                 onSubmit={handleContactSubmit} className="space-y-5 bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-8 rounded-3xl shadow-2xl"
              >
                 <div className="space-y-4">
                   <input type="text" name="name" placeholder="Callsign / Name" required className="w-full bg-[#020617] border border-slate-700 rounded-xl px-5 py-4 text-white font-medium focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all placeholder:text-slate-600" />
                   <input type="email" name="email" placeholder="Secure Email" required className="w-full bg-[#020617] border border-slate-700 rounded-xl px-5 py-4 text-white font-medium focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all placeholder:text-slate-600" />
                   <textarea name="message" rows="4" placeholder="Directive details..." required className="w-full bg-[#020617] border border-slate-700 rounded-xl px-5 py-4 text-white font-medium focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition-all placeholder:text-slate-600 resize-none"></textarea>
                 </div>
                 <button type="submit" className="w-full bg-indigo-600 text-white font-black uppercase tracking-wider py-4 rounded-xl hover:bg-indigo-500 transition-all shadow-[0_0_20px_rgba(79,70,229,0.3)] active:scale-95 flex justify-center items-center">
                    {contactStatus || 'Transmit Payload'}
                 </button>
              </motion.form>
           </div>
        </section>

      </main>

      {/* --- FOOTER & LEGAL DISCLAIMER --- */}
      <footer className="border-t border-slate-800/60 pt-12 pb-8 bg-[#020617] relative z-10">
        <div className="max-w-7xl mx-auto px-6">
          
          {/* Top Footer Row */}
          <div className="flex flex-col md:flex-row justify-between items-center gap-6 mb-8">
             <div className="flex items-center space-x-3">
                <LogoIcon className="w-8 h-8 opacity-50 grayscale" />
                <span className="font-bold text-slate-500 tracking-tight">Nawthviper Systems</span>
             </div>
             <div className="text-slate-600 text-sm font-medium flex items-center">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 mr-2"></span>
                Engineered for MetaTrader 5
             </div>
          </div>

          {/* Legal Disclaimer Box */}
          <div className="border-t border-slate-800/60 pt-8">
            <p className="text-xs leading-relaxed text-slate-400 text-justify md:text-left font-mono">
              <strong className="text-slate-200 uppercase tracking-wider">Risk Warning & Legal Disclaimer:</strong> Nawthviper Systems is a technology and software development provider, not a registered financial services provider, broker, or investment advisor. Trading foreign exchange (Forex), CFDs, synthetic indices, and cryptocurrencies on margin carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade, you should carefully consider your investment objectives, level of experience, and risk appetite. 
              <br /><br />
              The Nawthviper terminal is an execution tool that operates based on mathematical models and historical data structures. It is designed for educational and infrastructural purposes only. Past performance of any trading system or methodology is not necessarily indicative of future results. You could sustain a loss of some or all of your initial capital. No representation is being made that any account will or is likely to achieve profits or losses similar to those discussed on this website. By using this software, you acknowledge that you are trading at your own risk, and Nawthviper Systems, its founders, and affiliates assume no liability for any trading losses or software execution errors.
            </p>
          </div>

        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
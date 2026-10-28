import { Menu, ArrowRight } from "lucide-react";

export function TopBar({
  onLogo,
  onBack,
}: {
  onLogo: () => void;
  onBack: () => void;
}): JSX.Element {
  return (
    <header className="border-b border-white/10 bg-ink/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1180px] items-center justify-between px-5 py-5 sm:px-8 lg:px-10">
        <button
          onClick={onLogo}
          className="flex items-center gap-3"
          aria-label="Return to planner start"
        >
          <span className="grid h-8 w-8 place-items-center bg-gradient-to-br from-violet to-cyan text-xs font-extrabold text-black">
            CC
          </span>
          <span className="text-sm font-bold tracking-[0.14em]">
            COMPUTE <span className="text-violet">CAST</span>
          </span>
        </button>
        <nav className="hidden items-center gap-8 text-xs text-[#a7a0bb] md:flex">
          <button onClick={onBack} className="transition hover:text-white">
            Overview
          </button>
          <span>Estimator</span>
          <span>
            Model review <sup className="text-violet">soon</sup>
          </span>
        </nav>
        <button
          className="text-[#a7a0bb] md:hidden"
          aria-label="Open navigation"
        >
          <Menu size={20} />
        </button>
      </div>
    </header>
  );
}
function Landing({ onStart }: { onStart: () => void }): JSX.Element {
  return (
    <section className="relative min-h-screen overflow-hidden bg-black">
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            "radial-gradient(ellipse 42% 54% at 2% 30%, rgba(199,43,177,.82) 0%, rgba(80,19,105,.5) 38%, transparent 72%), radial-gradient(ellipse 64% 48% at 55% 84%, rgba(24,28,206,.82) 0%, rgba(22,7,123,.72) 42%, transparent 76%), radial-gradient(ellipse 50% 32% at 80% 108%, rgba(47,220,214,.78) 0%, rgba(8,79,148,.56) 34%, transparent 70%), radial-gradient(ellipse 35% 52% at 104% 48%, rgba(20,25,194,.9) 0%, rgba(10,5,72,.62) 58%, transparent 100%), linear-gradient(145deg,#09000e 0%,#00000a 43%,#02001d 70%,#02000b 100%)",
        }}
      />
      <div className="absolute -right-24 top-12 h-[42vh] w-[52vw] rounded-[45%] bg-black/75 blur-2xl" />
      <div className="relative mx-auto flex min-h-screen max-w-[1280px] flex-col px-6 pb-10 pt-8 sm:px-10 lg:px-16">
        <LandingNav onStart={onStart} />
        <div className="flex flex-1 items-center pb-20 pt-28 sm:pt-36">
          <div className="max-w-4xl">
            <h1 className="text-[clamp(3.5rem,10vw,8.8rem)] font-extrabold leading-[.88] tracking-[-.075em] text-[#eeeeee]">
              COMPUTE <span className="text-[#8e7cea]">CAST</span>
            </h1>
            <p className="mt-9 max-w-[610px] text-lg font-light leading-[1.28] text-[#e0dce8] sm:text-2xl">
              estimate GPU resources and recommend optimizations for model
              training
            </p>
            <p className="mt-5 max-w-[520px] text-sm leading-6 text-[#bdb7cc]">
              Know whether your model fits{" "}
              <strong className="font-semibold text-white">
                before you rent the GPUs
              </strong>
              . Get VRAM, runtime and cost estimates, then explore ways to
              improve your plan.
            </p>
            <div className="mt-10 flex flex-col items-start gap-4 sm:flex-row sm:items-center">
              <button
                onClick={onStart}
                className="group flex items-center gap-3 bg-white px-6 py-4 text-sm font-semibold text-black transition hover:bg-violet hover:text-white"
              >
                Open estimator{" "}
                <ArrowRight
                  size={17}
                  className="transition group-hover:translate-x-1"
                />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
function LandingNav({ onStart }: { onStart: () => void }): JSX.Element {
  return (
    <nav className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <span className="grid h-8 w-8 place-items-center border border-white/30 text-[10px] font-bold">
          CC
        </span>
        <span className="text-xs font-bold tracking-[.18em]">COMPUTE CAST</span>
      </div>
      <button
        onClick={onStart}
        className="border border-white/30 px-4 py-2 text-xs text-white transition hover:border-white hover:bg-white hover:text-black"
      >
        Launch tool
      </button>
    </nav>
  );
}
function HowItWorks({ onStart }: { onStart: () => void }): JSX.Element {
  const items = [
    {
      n: "01",
      title: "Upload your workload",
      body: "Upload PyTorch scripts and JSON or YAML configs, then review your inputs.",
      tag: "Python · JSON · YAML",
    },
    {
      n: "02",
      title: "Choose your hardware",
      body: "Pick a GPU from the list with VRAM and price shown, or enter your own machine.",
      tag: "GPU · VRAM · $/hr",
    },
    {
      n: "03",
      title: "See the fit, then optimize",
      body: "Check memory fit, compare hardware and benchmark when you need a measurement.",
      tag: "Fits · Tight · Exceeds",
    },
  ];
  return (
    <main className="bg-black px-6 pb-24 pt-20 sm:px-10 lg:px-16">
      <div className="mx-auto max-w-[1180px]">
        <p className="eyebrow text-[#8e7cea]">HOW IT WORKS</p>
        <h2 className="mt-3 max-w-2xl text-3xl font-semibold tracking-tight sm:text-4xl">
          Three steps from model to{" "}
          <span className="text-[#8e7cea]">compute plan</span>
        </h2>
        <div className="mt-10 grid gap-4 md:grid-cols-3">
          {items.map((i) => (
            <article
              key={i.n}
              className="border border-white/15 bg-white/[0.03] p-6"
            >
              <span className="text-xs font-bold text-[#8e7cea]">{i.n}</span>
              <h3 className="mt-4 text-lg font-semibold">{i.title}</h3>
              <p className="mt-3 text-sm leading-6 text-[#bdb7cc]">{i.body}</p>
              <p className="mt-5 inline-block border border-white/20 px-2 py-1 font-mono text-[11px] text-[#a7a0bb]">
                {i.tag}
              </p>
            </article>
          ))}
        </div>
        <div
          className="mt-10 border border-white/15 p-6"
          aria-label="Example result, illustrative"
        >
          <p className="eyebrow">EXAMPLE RESULT (ILLUSTRATIVE)</p>
          <div className="mt-4 h-4 border border-white/20">
            <div className="h-full w-[62%] bg-[#7258dd]" />
          </div>
          <div className="mt-2 flex justify-between text-xs text-[#a7a0bb]">
            <span>Peak VRAM</span>
            <span>Usable limit</span>
          </div>
        </div>
        <button
          onClick={onStart}
          className="mt-10 bg-white px-6 py-4 text-sm font-semibold text-black transition hover:bg-violet hover:text-white"
        >
          Open estimator
        </button>
      </div>
    </main>
  );
}
export function LandingPage({ onStart }: { onStart: () => void }): JSX.Element {
  return (
    <>
      <Landing onStart={onStart} />
      <HowItWorks onStart={onStart} />
    </>
  );
}

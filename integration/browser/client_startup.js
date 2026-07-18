const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export async function joinClientPair(
  joinClient,
  controlledArguments,
  opponentArguments,
  { staggerMilliseconds = 750, sleep = delay } = {},
) {
  const controlledPromise = joinClient(...controlledArguments);
  const opponentPromise = (async () => {
    await sleep(staggerMilliseconds);
    return joinClient(...opponentArguments);
  })();
  const [controlled, opponent] = await Promise.all([controlledPromise, opponentPromise]);
  return { controlled, opponent };
}

export async function joinConfiguredClients(
  joinClient,
  controlledArguments,
  opponentArguments,
  { humanOpponent = false, ...pairOptions } = {},
) {
  if (humanOpponent) {
    return { controlled: await joinClient(...controlledArguments), opponent: null };
  }
  return joinClientPair(joinClient, controlledArguments, opponentArguments, pairOptions);
}

export async function closeBrowserResources(browser, contexts) {
  await Promise.allSettled(contexts.map((context) => context.close()));
  await browser.close().catch(() => {});
}
